use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Character {
    pub name: String,
    pub ids: Vec<i16>,
    #[serde(default)]
    pub reference_animation_offset: u64,
    #[serde(default)]
    pub reference_animation_length: usize,
}

impl Character {
    pub const fn requires_reference_animation(&self) -> bool {
        self.reference_animation_offset != 0 || self.reference_animation_length != 0
    }

    pub fn has_id(&self, id: i16) -> bool {
        self.ids.contains(&id)
    }
}

#[derive(Debug, Clone)]
pub struct CharacterMapping {
    pub from: Character,
    pub to: Character,
    pub skeleton_mapping: Vec<Option<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct Schema {
    characters: Vec<Character>,
    mappings: HashMap<String, HashMap<String, Vec<isize>>>,
}

impl Schema {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    fn find_character(&self, id: i16) -> Option<&Character> {
        self.characters.iter().find(|character| character.ids.contains(&id))
    }

    pub fn get_mapping(&self, from: i16, to: i16) -> Result<CharacterMapping> {
        let Some(from_character) = self.find_character(from) else { bail!("Character {} not found", from) };
        let Some(to_character) = self.find_character(to) else { bail!("Character {} not found", to) };

        let from_name = from_character.name.as_str();
        let to_name = to_character.name.as_str();
        let Some(from_map) = self.mappings.get(from_name) else { bail!("No mappings found for {}", from_name) };
        match from_map.get(to_name) {
            Some(mapping) => Ok(CharacterMapping {
                from: from_character.clone(),
                to: to_character.clone(),
                skeleton_mapping: mapping.iter().map(|&bone_id| (bone_id >= 0).then(|| bone_id as usize)).collect(),
            }),
            None => bail!("No mapping found from {} to {}", from_name, to_name),
        }
    }
}