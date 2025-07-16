conda env create -f environment.yml
conda activate claspenv
<!-- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html -->
conda install pytorch=2.6.0 torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu122.html


# TO DO LIST IN REPO

Code first pass
- [x] Preprocessing scripts
- [x] Training script
- [ ] Sim matrix + emb script
- [ ] Quick sim script
- [ ] Retreval script
- [ ] Download script

Admin / docs
- [ ] README.md
- [ ] Preprocessing.md
- [ ] Training.md
- [ ] Sim_matrix.md
- [ ] Quick_sim.md
- [ ] Retrieval.md
- [ ] Upload_data.md

Code final pass
- [ ] Preprocessing scripts
- [ ] Training script
- [ ] Sim matrix + emb script
- [ ] Quick sim script
- [ ] Retreval script
- [ ] Download script
