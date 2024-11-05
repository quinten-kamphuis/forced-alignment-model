# Forced Audio-Text Alignment

Generate precise word-level timings from audio and text input. Built using torchaudio's MMS model.

[![Replicate](https://replicate.com/quinten-kamphuis/forced-alignment/badge)](https://replicate.com/quinten-kamphuis/forced-alignment)

## Overview

This model aligns audio with text to generate word-level timings, useful for:
- Generating accurate subtitles/captions
- Creating word-level audio segmentation
- Synchronizing text with audio

Try it out on [Replicate](https://replicate.com/quinten-kamphuis/forced-alignment)!

## Development

1. Install Cog:
```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog
```

2. Run predictions:
```bash
cog predict -i audio=@audio.mp3 -i script="Your transcript here"
```

3. Push to Replicate:
```bash
cog push r8.im/username/forced-alignment
```

## License

MIT License
