def printing_model(model, model_name):
    tot_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-------------------------------------------")
    print(f"{model_name} uploaded correctly!")
    print(f"Total parameters: {tot_params:,}")
    print(f"Trainable parameters: {train_params:,}")
    print(f"Non-trainable parameters: {tot_params - train_params:,}")
    print("-------------------------------------------")

def printing_train(model_name):
    print(f"\n--------------------------------------------------\nStarted training of {model_name}\n--------------------------------------------------")

def printing_test(model_name):
    print(f"\n--------------------------------------------------\n{model_name} correctly uploaded for eval\n--------------------------------------------------")
    print(f"\n--------------------------------------------------\nStarted testing of {model_name}\n--------------------------------------------------")