use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;

use crate::animation::Animation;
use crate::schema::CharacterMapping;
use crate::skeleton::Skeleton;

#[derive(Debug)]
pub struct AnimationConverter {
    mapping: CharacterMapping,
    input_animation: Animation,
    reference_animation_data: Vec<u8>,
}

impl AnimationConverter {
    pub fn new(mapping: CharacterMapping, input_animation: Animation) -> Self {
        Self {
            mapping,
            input_animation,
            reference_animation_data: Vec::new(),
        }
    }

    pub fn load_reference_animation(&mut self, path: &Path) -> Result<()> {
        let to = &self.mapping.to;

        if !to.requires_reference_animation() {
            // nothing we need to do here
            return Ok(());
        }

        let mut file = File::open(path)?;
        if to.reference_animation_offset > 0 {
            file.seek(SeekFrom::Start(to.reference_animation_offset))?;
        }

        self.reference_animation_data.clear();
        if to.reference_animation_length == 0 {
            file.read_to_end(&mut self.reference_animation_data)?;
        } else {
            self.reference_animation_data.resize(to.reference_animation_length, 0);
            file.read_exact(&mut self.reference_animation_data)?;
        }

        Ok(())
    }

    pub fn convert(&self, output_skeleton: &Skeleton, output_path: &Path, use_model_translations: bool) -> Result<()> {
        let mut file = File::create(output_path)?;
        self.input_animation.write_for_skeleton(output_skeleton, self.mapping.skeleton_mapping.as_slice(), use_model_translations, &mut file)?;
        file.write_all(&self.reference_animation_data)?;

        Ok(())
    }
}