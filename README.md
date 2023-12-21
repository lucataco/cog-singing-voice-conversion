# amphion/singing_voice_conversion Cog model

This is an implementation of [amphion/singing_voice_conversion](https://huggingface.co/amphion/singing_voice_conversion) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Setup
Follow the detailed instructions on the huggingface model page. These general steps are:
- git install lfs
- git clone https://huggingface.co/amphion/singing_voice_conversion
- git clone https://huggingface.co/amphion/BigVGAN_singing_bigdata
- git clone https://github.com/open-mmlab/Amphion.git
- cd Amphion && mkdir -p ckpts/svc
- cd Amphion/pretrained/contentvec  && wget checkpoint_best_legacy_500.pt
- mv singing_voice_conversion/vocalist_l1_contentvec+whisper Amphion/ckpts/svc/vocalist_l1_contentvec+whisper
- mv BigVGAN_singing_bigdata/bigvgan_singing Amphion/pretrained/bigvgan

## Basic Usage

Run a prediction

    cog predict -i source_audio=@demo.wav

## Output

Default is set to take in Adeles song(demo.wav), and convert it to have Taylor Swift sing it instead

See the audio file: output.wav
