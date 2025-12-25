use speech_rs::asr;

fn main() {
    let  engine = asr::paraformer::ParaformerEngine::new(
        "paraformer_weights/model.int8.onnx",
        "paraformer_weights/tokens.txt",
        "paraformer_weights/am.mvn",
    );
    match engine {
        Ok(mut engine) => {
            match engine.transcribe_file("paraformer_weights/zh-en.wav"){
                Ok(text) => {
                    println!("Transcription: {}", text);
                }
                Err(e) => {
                    eprintln!("Failed to transcribe audio file: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to initialize Paraformer engine: {}", e);
        }
    }
}
