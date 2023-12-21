# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import subprocess

SUPPORTED_TARGET_SINGERS = {
    "Adele": "vocalist_l1_Adele",
    "Beyonce": "vocalist_l1_Beyonce",
    "Bruno Mars": "vocalist_l1_BrunoMars",
    "John Mayer": "vocalist_l1_JohnMayer",
    "Michael Jackson": "vocalist_l1_MichaelJackson",
    "Taylor Swift": "vocalist_l1_TaylorSwift",
    "Jacky Cheung 张学友": "vocalist_l1_张学友",
    "Jian Li 李健": "vocalist_l1_李健",
    "Feng Wang 汪峰": "vocalist_l1_汪峰",
    "Faye Wong 王菲": "vocalist_l1_王菲",
    "Yijie Shi 石倚洁": "vocalist_l1_石倚洁",
    "Tsai Chin 蔡琴": "vocalist_l1_蔡琴",
    "Ying Na 那英": "vocalist_l1_那英",
    "Eason Chan 陈奕迅": "vocalist_l1_陈奕迅",
    "David Tao 陶喆": "vocalist_l1_陶喆",
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        source_audio: Path = Input(description="Input source audio file"),
        target_singer: str = Input(
            description="Target singer to convert audio to",
            default="Taylor Swift",
            choices=SUPPORTED_TARGET_SINGERS.keys(),
        ),
        pitch_shift_control: str = Input(
            description="Pitch shift control",
            default="Auto Shift",
            choices=["Auto Shift", "Key Shift"],
        ),
        key_shift_mode: int = Input(
            description="Key shift values",
            default=0, ge=-6, le=6
        ),
        diffusion_inference_steps: int = Input(
            description="Diffusion inference steps",
            default=1000, ge=0, le=1000
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Change to the working directory
        os.chdir("Amphion")

        # Clean past runs
        input_audio_folder = "/tmp/input_audio"
        os.system(f"rm -rf {input_audio_folder}")
        os.system(f"rm -rf result")
        # Copy input audio file to tmp folder  
        os.system(f"mkdir -p {input_audio_folder}")
        os.system(f"cp {source_audio} {input_audio_folder}/source.wav")
        # Create result folder
        os.system(f"mkdir -p result")

        # Target Singer
        target_singer = SUPPORTED_TARGET_SINGERS[target_singer]
        if pitch_shift_control == "Auto Shift":
            key_shift = "autoshift"
        else:
            key_shift = key_shift_mode

        print(input_audio_folder)
        print(target_singer)
        print(key_shift)
        # Run the conversion script
        subprocess.run([
            "bash", "egs/svc/MultipleContentsSVC/run.sh",
            "--stage", "3",
            "--gpu", "0",
            "--config", "ckpts/svc/vocalist_l1_contentvec+whisper/args.json",
            "--infer_expt_dir", "ckpts/svc/vocalist_l1_contentvec+whisper",
            "--infer_output_dir", "result",
            "--infer_source_audio_dir", input_audio_folder,
            "--infer_vocoder_dir", "pretrained/bigvgan",
            "--infer_target_speaker", target_singer,
            "--infer_key_shift", key_shift,
            "--diffusion_inference_steps", str(diffusion_inference_steps)
        ])

        # Search for an audio file in the folder result and return it
        output_audio_folder = "/src/Amphion/result/source/"
        for file in os.listdir(output_audio_folder):
            if file.endswith(".wav"):
                output_path = os.path.join(output_audio_folder, file)
                print(output_path)
                return Path(output_path)
