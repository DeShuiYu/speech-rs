use hound::WavReader;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor};

pub const SAMPLE_RATE: usize = 16000;
/// Mono
pub const CHANNELS: usize = 1;
/// Bits per sample
pub const BITS_PER_SAMPLE: usize = 16;

pub struct ParaformerEngine {
    session: Session,
    token_map: HashMap<i32, String>,
    neg_mean: Vec<f32>, // CMVN 均值
    inv_std: Vec<f32>,  // CMVN 方差倒数
}

impl ParaformerEngine {
    pub fn new(model_path: &str, token_path: &str, cmvn_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        let token_map = Self::load_tokens(token_path)?;
        let (neg_mean, inv_std) = Self::load_cmvn(cmvn_path)?;
        Ok(Self {
            session,
            token_map,
            neg_mean,
            inv_std,
        })
    }

    pub fn transcribe_file(&mut self, wav_path: &str) -> anyhow::Result<String> {
        let wav_bytes = std::fs::read(wav_path)?;
        self.transcribe_bytes(&wav_bytes)
    }

    fn transcribe_bytes(&mut self, wav_bytes: &[u8]) -> anyhow::Result<String> {
        let samples = self.parse_wav_bytes(wav_bytes)?;
        self.transcribe(&samples)
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        if samples.is_empty() {
            return Err(anyhow::anyhow!("输入的音频数据为空"));
        }
        let (features, feat_len) = self.extract_features(samples)?;

        // run inference
        let ids = self.run_inference(&features, feat_len)?;

        let text = self.decode(&ids);
        Ok(text)
    }

    fn run_inference(&mut self, features: &[f32], feat_len: i32) -> anyhow::Result<Vec<i32>> {
        let token_size = 560usize;
        let batch = 1usize;
        let expected = batch * (feat_len as usize) * token_size;
        if features.len() != expected {
            return Err(anyhow::anyhow!(
                "features 长度不匹配期望值: {} != {}",
                features.len(),
                expected
            ));
        }

        // Create input as (shape, Vec<f32>) to satisfy ort's OwnedTensorArrayData impls
        let input_shape = vec![batch, feat_len as usize, token_size];
        let input_vec = features.to_vec();

        // length 输入（Paraformer 模型通常为 int64）
            // length 输入（Paraformer 模型通常为 int64）
            // Use a (shape, Vec<T>) form so it satisfies the crate's OwnedTensorArrayData impls
        let len_shape = vec![1usize];
        let len_vec = vec![feat_len as i32];

        // 直接从 owned Array 创建 Tensor
        let input_tensor: Tensor<f32> = Tensor::from_array((input_shape, input_vec))?;
        let len_tensor: Tensor<i32> = Tensor::from_array((len_shape, len_vec))?;

        // 运行推理（按顺序输入）
        let outputs = self.session.run(ort::inputs![input_tensor, len_tensor])?;

        // 如果模型输入有名称（推荐打印 self.session.inputs 查看），可改成：
        // let outputs = self.session.run(ort::inputs!["speech" => input_tensor, "speech_lengths" => len_tensor])?;

        let (shape, raw_data) = outputs[0].try_extract_tensor::<f32>()?;

        if shape.len() < 3 {
            return Err(anyhow::anyhow!("输出结果维度异常: {:?}", shape));
        }
        let steps = shape[1] as usize;
        let out_token_size = shape[2] as usize;
        Ok(Self::get_token_ids(raw_data, steps, out_token_size))
    }

    fn parse_wav_bytes(&self, wav_bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        let cur = Cursor::new(wav_bytes);
        let mut reader = WavReader::new(cur)?;
        let spec = reader.spec();
        // We will re-sample / re-format if needed is not implemented here — assume input is 16kHz mono 16-bit
        let samples: Result<Vec<f32>, _> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                if bits == 16 {
                    Ok(reader
                        .samples::<i16>()
                        .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
                        .collect())
                } else {
                    // other bit depths not implemented
                    Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "unsupported bit depth",
                    ))
                }
            }
            hound::SampleFormat::Float => {
                Ok(reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect())
            }
        };
        Ok(samples?)
    }

    fn decode(&self, ids: &[i32]) -> String {
        let mut out = String::new();
        for &idx in ids {
            if let Some(word) = self.token_map.get(&idx) {
                if word == "<blank>" || word == "<s>" || word == "</s>" || word == "<unk>" {
                    out.push(' ');
                } else if word.ends_with("@@") {
                    let w = word.replace("@@", "");
                    out.push_str(&w);
                } else {
                    out.push_str(word);
                    out.push(' ');
                }
            }
        }
        out
    }

    fn get_token_ids(token_scores: &[f32], steps: usize, token_size: usize) -> Vec<i32> {
        let mut ids = Vec::with_capacity(steps);
        for t in 0..steps {
            let start = t * token_size;
            let end = start + token_size;
            if end > token_scores.len() {
                break;
            }

            let mut max_idx: usize = 0;
            let mut max_val: f32 = -std::f32::MAX;
            for (i, &v) in token_scores[start..end].iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            ids.push(max_idx as i32);
        }
        ids
    }
    fn hamming_window(&self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let n = i as f32;
                let m = len as f32;
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * n / (m - 1.0)).cos()
            })
            .collect()
    }

    fn pre_emphasis(&self, input: &[f32], coeff: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        if input.is_empty() {
            return out;
        }
        out.push(input[0]);
        for i in 1..input.len() {
            out.push(input[i] - coeff * input[i - 1]);
        }
        out
    }

    fn next_power_of_two(&self, n: usize) -> usize {
        n.next_power_of_two()
    }

    /// compute mel filters (triangular) — returns Vec< Vec<f32> > of shape mel_bins x (fft_size/2+1)
    fn mel_filters(
        &self,
        sample_rate: usize,
        fft_size: usize,
        mel_bins: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let fmax = if fmax <= 0.0 {
            sample_rate as f32 / 2.0
        } else {
            fmax
        };
        let fmin = fmin;

        fn hz_to_mel(hz: f32) -> f32 {
            2595.0 * (1.0 + hz / 700.0).log10()
        }
        fn mel_to_hz(mel: f32) -> f32 {
            700.0 * (10f32.powf(mel / 2595.0) - 1.0)
        }

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mels: Vec<f32> = (0..(mel_bins + 2))
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((mel_bins + 1) as f32))
            .collect();
        let hzs: Vec<f32> = mels.iter().map(|m| mel_to_hz(*m)).collect();
        let bins: Vec<usize> = hzs
            .iter()
            .map(|hz| ((hz / (sample_rate as f32)) * (fft_size as f32)).floor() as usize)
            .collect();

        let mut filters = vec![vec![0f32; fft_size / 2 + 1]; mel_bins];
        for m in 0..mel_bins {
            let f_m_minus = bins[m];
            let f_m = bins[m + 1];
            let f_m_plus = bins[m + 2];
            if f_m_minus >= f_m || f_m >= f_m_plus {
                continue;
            }
            for k in f_m_minus..f_m {
                if k < filters[m].len() {
                    filters[m][k] = (k as f32 - f_m_minus as f32) / (f_m as f32 - f_m_minus as f32);
                }
            }
            for k in f_m..f_m_plus {
                if k < filters[m].len() {
                    filters[m][k] = (f_m_plus as f32 - k as f32) / (f_m_plus as f32 - f_m as f32);
                }
            }
        }
        filters
    }

    /// compute filter bank, return features [num_frames][mel_bins]
    pub fn compute_filter_bank(
        &self,
        samples: &[f32],
        sample_rate: usize,
        mel_bins: usize,
    ) -> (Vec<Vec<f32>>, usize) {
        const FRAME_LEN: usize = 400; // 25ms @16k
        const FRAME_SHIFT: usize = 160; // 10ms

        let fft_size = self.next_power_of_two(512);

        let window = self.hamming_window(FRAME_LEN);
        let mel_filters = self.mel_filters(
            sample_rate,
            fft_size,
            mel_bins,
            0.0,
            sample_rate as f32 / 2.0,
        );

        let emphasized = self.pre_emphasis(samples, 0.97);
        let num_samples = emphasized.len();
        if num_samples < FRAME_LEN {
            return (vec![], 0);
        }
        let num_frames = (num_samples - FRAME_LEN) / FRAME_SHIFT + 1;

        let mut features = vec![vec![0f32; mel_bins]; num_frames];

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);

        for i in 0..num_frames {
            let start = i * FRAME_SHIFT;
            let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
            for j in 0..fft_size {
                if j < FRAME_LEN {
                    buffer[j] = Complex {
                        re: emphasized[start + j] * window[j],
                        im: 0.0,
                    };
                } else {
                    buffer[j] = Complex::new(0.0, 0.0);
                }
            }
            fft.process(&mut buffer);
            // compute power spectrum and mel energies
            for k in 0..mel_bins {
                let mut sum = 0f32;
                for j in 0..(fft_size / 2 + 1) {
                    let w = mel_filters[k][j];
                    if w > 0.0 {
                        let r = buffer[j].re;
                        let im = buffer[j].im;
                        let power = r * r + im * im;
                        sum += power * w;
                    }
                }
                if sum < 1e-7 {
                    sum = 1e-7;
                }
                features[i][k] = sum.ln();
            }
        }

        (features, num_frames)
    }

    /// apply LFR
    pub fn apply_lfr(
        &self,
        inputs: &[Vec<f32>],
        num_frames: usize,
        input_dim: usize,
        lfr_m: usize,
        lfr_n: usize,
    ) -> (Vec<Vec<f32>>, usize) {
        if num_frames < lfr_m {
            return (vec![], 0);
        }
        let out_frames = (num_frames - lfr_m) / lfr_n + 1;
        let out_dim = input_dim * lfr_m;
        let mut output = vec![vec![0f32; out_dim]; out_frames];
        for i in 0..out_frames {
            let start_frame = i * lfr_n;
            for j in 0..lfr_m {
                let src_idx = start_frame + j;
                let dest_pos = j * input_dim;
                output[i][dest_pos..dest_pos + input_dim].copy_from_slice(&inputs[src_idx]);
            }
        }
        (output, out_frames)
    }

    /// extract features: wrapper similar to Go
    pub fn extract_features(&self, samples: &[f32]) -> Result<(Vec<f32>, i32), anyhow::Error> {
        const MEL_BINS: usize = 80;
        const LFR_M: usize = 7;
        const LFR_N: usize = 6;

        let (fbank, num_frames) = self.compute_filter_bank(samples, SAMPLE_RATE, MEL_BINS);
        if num_frames == 0 {
            return Err(anyhow::anyhow!("FBank特征提取失败: 帧数小于 1"));
        }
        let (lfr_data, lfr_frames) = self.apply_lfr(&fbank, num_frames, MEL_BINS, LFR_M, LFR_N);
        if lfr_frames == 0 {
            return Err(anyhow::anyhow!("LFR特征提取失败: 帧数小于 1"));
        }
        // flatten
        let mut flattened = Vec::with_capacity(lfr_frames * MEL_BINS * LFR_M);
        for frame in lfr_data.iter() {
            flattened.extend_from_slice(frame);
        }
        Ok((flattened, lfr_frames as i32))
    }

    fn load_cmvn(cmvn_path: &str) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let file = File::open(cmvn_path)?;
        let reader = BufReader::new(file);
        let mut neg_mean: Option<Vec<f32>> = None;
        let mut inv_std: Option<Vec<f32>> = None;
        for line in reader.lines() {
            let line = line?.trim().to_string();
            if !line.starts_with("<LearnRateCoef>") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() <= 4 {
                continue;
            }
            let data_parts = &parts[3..&parts.len() - 1];
            let mut values = Vec::with_capacity(data_parts.len());
            for v in data_parts {
                if let Ok(fv) = v.parse::<f32>() {
                    values.push(fv);
                }
            }
            if neg_mean.is_none() {
                neg_mean = Some(values);
            } else {
                inv_std = Some(values);
                break;
            }
        }
        match (neg_mean, inv_std) {
            (Some(nm), Some(is)) => Ok((nm, is)),
            _ => Err(anyhow::anyhow!("Failed to load CMVN data")),
        }
    }

    fn load_tokens(token_path: &str) -> anyhow::Result<HashMap<i32, String>> {
        let file = File::open(token_path)?;
        let reader = BufReader::new(file);
        let mut token_map = HashMap::new();
        for line in reader.lines() {
            let line = line?.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<i32>() {
                    token_map.insert(id, token);
                }
            }
        }
        Ok(token_map)
    }
}

#[cfg(test)]
mod tests {
    use super::ParaformerEngine;

    /***
         * cargo test --color=always --package speech-rs --lib asr::paraformer::tests::test_paraformer -- --nocapture                                                                                             Desktop/speech-rs (main ⚡) MacBook-Pro-M1
            running 1 test
            Transcription: yesterday was 星 期 一 today is tuesday 明 天 是 星 期 三
            test asr::paraformer::tests::test_paraformer ... ok
            test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.59s
         */

    #[test]
    fn test_paraformer() {
        let engine = ParaformerEngine::new(
            "paraformer_weights/model.int8.onnx",
            "paraformer_weights/tokens.txt",
            "paraformer_weights/am.mvn",
        );
        match engine {
            Ok(mut engine) => match engine.transcribe_file("paraformer_weights/zh-en.wav") {
                Ok(text) => {
                    println!("Transcription: {}", text);
                }
                Err(e) => {
                    eprintln!("Failed to transcribe audio file: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Failed to initialize Paraformer engine: {}", e);
            }
        }
    }
}
