use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{arg, command, value_parser, ArgAction};

use sh2_anim_converter::animation::Animation;
use sh2_anim_converter::convert::AnimationConverter;
use sh2_anim_converter::schema::Schema;
use sh2_anim_converter::skeleton::Skeleton;

fn main() -> Result<()> {
    let matches = command!()
        .arg(
            arg!(-s --schema <SCHEMA> "The schema.toml file describing how to map between characters. If not provided, the file is expected to be in the current directory.")
                .value_parser(value_parser!(PathBuf))
                .default_value("schema.toml")
        )
        .arg(
            arg!(-a --"reference-animation" <REFERENCE_ANIMATION> "An animation file for the output character to use as a reference for the output animation. This may be required depending on the character; some characters require data from an existing animation.")
                .value_parser(value_parser!(PathBuf))
        )
        .arg(
            arg!(-t --"model-translation" "Instead of using the translations from the input animation directly, use translations from the output model, scaled based on the magnitude of the animation translations. This can help avoid distorted proportions when characters are different sizes.")
                .action(ArgAction::SetTrue)
        )
        .arg(arg!(<INPUT_MODEL> "The model file for the character whose animation will be converted.").value_parser(value_parser!(PathBuf)))
        .arg(arg!(<INPUT_FILE> "The animation file to convert.").value_parser(value_parser!(PathBuf)).value_parser(value_parser!(PathBuf)))
        .arg(arg!(<OUTPUT_MODEL> "The model file for the character to convert the animation to.").value_parser(value_parser!(PathBuf)))
        .arg(arg!(<OUTPUT_FILE> "The output file to write the converted animation to.").value_parser(value_parser!(PathBuf)))
        .get_matches()
        ;

    let schema_path = matches.get_one::<PathBuf>("schema").unwrap();
    let reference_animation_path = matches.get_one::<PathBuf>("reference-animation");
    let input_model_path = matches.get_one::<PathBuf>("INPUT_MODEL").unwrap();
    let input_animation_path = matches.get_one::<PathBuf>("INPUT_FILE").unwrap();
    let output_model_path = matches.get_one::<PathBuf>("OUTPUT_MODEL").unwrap();
    let output_animation_path = matches.get_one::<PathBuf>("OUTPUT_FILE").unwrap();
    let use_model_translations = matches.get_flag("model-translation");

    let schema = Schema::load(schema_path).context("schema.toml")?;

    let input_skeleton = Skeleton::read_from_model(File::open(input_model_path).context("Input model")?)?;
    let output_skeleton = Skeleton::read_from_model(File::open(output_model_path).context("Output model")?)?;

    let mapping = schema.get_mapping(input_skeleton.character_id, output_skeleton.character_id)?;

    let input_animation = Animation::read(File::open(input_animation_path).context("Input animation")?, &input_skeleton)?;

    let mut converter = AnimationConverter::new(mapping, input_animation);
    if let Some(reference_animation_path) = reference_animation_path {
        converter.load_reference_animation(reference_animation_path)?;
    }
    converter.convert(&output_skeleton, output_animation_path, use_model_translations)
}
