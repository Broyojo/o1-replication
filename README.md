# o1 Replication

an effort to replicate o1

## TODO:
- clean up code and separate data from code
- make all the scripts use CLI args and have a main scripts/ folder. separate the project into:

scripts/
src/
|- rl/
|- sft/
|- verl/
|- common/
    |- grading/

and have proper python modules and everything. also, use classes to make things more modular and safe. should make the research more replicable and reliable overall. probably make a general pipeline for generating SFT data, training on it, etc. also make pipeline for the RL being done, with many reward functions for different domains and such