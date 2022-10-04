from lstm_ae import LSTMAE
from trainer import *

if __name__ == "__main__":
    # Load train, val and test datasets
    train_ds, val_ds, test_ds, target_len, num_features, scaler = load_train_val_test(path="../Data/20220823",
                                                                                      remove_features=("SPD", "ANG"),
                                                                                      transform=True)
    model = LSTMAE(seq_len=target_len, n_features=num_features, embedding_dim=128, num_layers=2)
    name = f"{model.name}_{model.num_layers}"

    # instantiate trainer and train
    trainer = Trainer(log_dir="../training_logs")
    hist = trainer.fit(model, train_ds, val_ds, num_epochs=50, learning_rate=1e-03)
    trainer.plot_loss_history(history=hist, filename="loss")

    forecast, _losses = trainer.predict(model, model_path=f"../training_logs/{name}.pth", dataset=test_ds)

    trainer.plot_prediction(dataset=test_ds, model=model, model_path=f"../training_logs/{name}.pth",
                            scaler=scaler, show_plot=False)
