Research
- Slerp and Linear interpolation review
    - Implement a few more times to get a deeper understanding
- Stable audio open review - Review the stable audio open stack of papers
- 4 anchor system
- Review sampler system - pingpong

Experiments
- Use Init Audio param - See how input audio can be SLERPED
- FINE TUNE ON DUBSTEP: Work with noel on this dataset
- Fine tune on Voice-Text dataset with Labels - duration might be a problem, maybe could condition, but might be hard.
    - Model explicitly says its not for speech, but it has learned audio, 
    - use VCTK - it will have metadata, and text, see if you can include in a format like TEXT | META or something
    - Expensive experiment - need to research the training time for a fine tune


UI
- LLM 
    - Have an XY grid
    - Interpolate between 4 dims
- This should be a web system: use FastAPI + HTML serving - much more flexable: need to plan out UI completly before building with LLM


Config
- Move to a config.yaml file just to make setup easier
- Add presets
- Add model loader - be able to select a model you trained


Prompt Morph Synthesis


NOTES
- I need to find a way to nicely name files - maybe just let the user do it.