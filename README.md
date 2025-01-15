# sh2-anim-converter

A tool to convert Silent Hill 2 animation files for one character to another character. Created for my
[Heaven's Knife](https://github.com/descawed/sh2-heavens-knife) mod to allow James and Maria to use each other's
weapon animations.

## Usage

**Note**: This has only been tested with player character weapon animations.

The tool converts an animation file for the *input* character to be usable by the *output* character. View help with
`sh2-anim-converter -h `. Example usage assuming the executable and schema.toml are in the game directory:

```
sh2-anim-converter -a data/chr2/mar/xmar_wpcolt.anm data/chr/jms/hll_jms.mdl data/chr/jms/jms_wphand.anm data/chr2/mar/lxx_mar.mdl data/chr2/mar/xmar_wphand.anm
```

* `-a data/chr2/mar/xmar_wpcolt.anm`: This specifies a reference animation for the output character. This is necessary
  for Maria's weapon animations because she has extra non-weapon animations at the end of all her weapon animation
  files that need to be copied over. However, there's other useful information that could be extracted from an existing
  animation file, which I may add support for in the future.
* `data/chr/jms/hll_jms.mdl`: The model file for the input character. The animation skeleton is extracted from this file.
* `data/chr/jms/jms_wphand.anm`: The animation for the input character that will be converted.
* `data/chr2/mar/lxx_mar.mdl`: The model file for the output character. The animation skeleton is extracted from this
  file, as well as the default transforms for each bone. For any output bones that aren't mapped to an input bone (see
  [Schema](#Schema)), the default transform will be used instead.
* `data/chr2/mar/xmar_wphand.anm`: Path to write the converted animation that can be used by the output character. Will
  be created if it doesn't exist.

Note that when converting Maria's animations to James, you need to truncate her animation at the reference animation
offset (see [Schema/Characters](#characters)) prior to conversion; the tool doesn't do this automatically yet.

## Schema

The schema, in schema.toml, describes how to map one character's skeleton to another. Only mappings between James and
Maria are provided. There are only two top-level keys, `characters` and `mappings`.

### characters

An array of characters for whom mappings are defined. Each character has the following properties:

* `name`: (required) A string name identifying the character. This is how the character is referred to in the
  `mappings` section.
* `ids`: (required) An array of ID numbers for this character. The ID number from the model files provided to the
  command will be used to look up the appropriate characters in the schema. This is an array primarily for James, who
  is the only character I know of with two interchangeable IDs.
* `reference_animation_offset`: (optional) If the character needs data from a reference animation, this specifies the
  offset in bytes to start reading from within the reference animation.
* `reference_animation_length`: (optional) If the character needs data from a reference animation, this specifies the
  length in bytes of the data to read from the reference animation. If an offset is specified without a length, data
  will be read from the offset to EOF.

### mappings

The `mapppings` section defines mappings from bones in an input character's skeleton to bones in an output character's
skeleton. Each entry in the mappings table should have a name of the form `input.output`, where `input` is the name
defined in the `characters` section for the input character, and `output` is the name defined in the `characters`
section for the output character. For example, the mapping from James' skeleton to Maria's is named `james.maria`.

Each mapping is a simple array, where the *n*th element in the array defines the index of the bone in the input
character's skeleton whose transform should be used for the *n*th bone in the output character's skeleton. A value of -1
indicates that there is no analogous bone in the input skeleton. In that case, the default transform from the model file
will be used. If an output bone is mapped to an input bone, but that input bone's parent bone is not mapped to any
output bone, the input bone's transform and all unmapped parent transforms will be recursively composed into a single
transform which will be used for the output bone.