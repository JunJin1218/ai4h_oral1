from assessor.model import AssessorConfig, AssessorMLP, save_assessor

def main():
    cfg = AssessorConfig(embedding_size=1280, hidden_sizes=(1024, 256), dropout=0.2)
    model = AssessorMLP(cfg).cuda()
    save_assessor(model, cfg, out_dir="assessor/model")

if __name__ == "__main__":
    main()