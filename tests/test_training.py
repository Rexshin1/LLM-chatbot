import os
import shutil
import math
import pytest
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import Trainer, CausalLMDataset

VOCAB_PATH = "data/tokenizer/vocab.json"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
TEST_CHECKPOINT_DIR = "checkpoints_test"

@pytest.fixture
def setup_components():
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(VOCAB_PATH)
    vocab_size = len(tokenizer.vocab)
    
    model_config = {
        "vocab_size": vocab_size,
        "d_model": 64,
        "num_heads": 2,
        "num_layers": 2,
        "max_seq_len": 32,
        "dropout": 0.1
    }
    model = DecoderOnlyTransformer(**model_config)
    
    train_dataset = CausalLMDataset(TRAIN_PATH, tokenizer, seq_len=16)
    val_dataset = CausalLMDataset(VAL_PATH, tokenizer, seq_len=16)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    return {
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "model_config": model_config
    }

def test_training_dataset(setup_components):
    """Verify that dataset produces correct input and target shapes."""
    dataset = setup_components["train_dataset"]
    x, y = dataset[0]
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (16,)
    assert y.shape == (16,)

def test_dataloader(setup_components):
    """Verify DataLoader outputs batches of correct shapes."""
    loader = setup_components["train_loader"]
    x, y = next(iter(loader))
    
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)

def test_single_training_step(setup_components):
    """Verify a single training step executes, computes loss, does backward and updates parameters."""
    model = setup_components["model"]
    optimizer = setup_components["optimizer"]
    loader = setup_components["train_loader"]
    
    params_before = [p.clone() for p in model.parameters() if p.requires_grad]
    
    model.train()
    x, y = next(iter(loader))
    logits, loss = model(x, labels=y)
    
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    params_after = [p.clone() for p in model.parameters() if p.requires_grad]
    any_diff = False
    for p_before, p_after in zip(params_before, params_after):
        if not torch.equal(p_before, p_after):
            any_diff = True
            break
    assert any_diff, "Model parameters were not updated after optimizer step!"

def test_validation_loop(setup_components):
    """Verify trainer validation loop runs successfully."""
    model = setup_components["model"]
    optimizer = setup_components["optimizer"]
    train_loader = setup_components["train_loader"]
    val_loader = setup_components["val_loader"]
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_clip=1.0,
        checkpoint_dir=TEST_CHECKPOINT_DIR
    )
    
    val_loss = trainer.validate()
    assert isinstance(val_loss, float)
    assert val_loss > 0
    assert not math.isnan(val_loss)

def test_checkpoint_save_load_resume(setup_components):
    """Verify that checkpoint can be saved, loaded, and training resumed from it."""
    model = setup_components["model"]
    optimizer = setup_components["optimizer"]
    train_loader = setup_components["train_loader"]
    val_loader = setup_components["val_loader"]
    model_config = setup_components["model_config"]
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_clip=1.0,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        model_config=model_config
    )
    
    train_loss = trainer.train_epoch(0)
    val_loss = trainer.validate()
    checkpoint_file = "test_model.pt"
    
    trainer.save_checkpoint(0, train_loss, val_loss, filename=checkpoint_file)
    checkpoint_path = os.path.join(TEST_CHECKPOINT_DIR, checkpoint_file)
    assert os.path.exists(checkpoint_path)
    
    new_model = DecoderOnlyTransformer(**model_config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    
    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_clip=1.0,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        model_config=model_config
    )
    
    checkpoint_data = new_trainer.load_checkpoint(checkpoint_path)
    
    assert new_trainer.start_epoch == 1
    assert new_trainer.best_val_loss == val_loss
    assert checkpoint_data["epoch"] == 0
    assert checkpoint_data["train_loss"] == train_loss
    assert checkpoint_data["validation_loss"] == val_loss
    assert checkpoint_data["model_config"] == model_config
    
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
        
    new_trainer.train(epochs=2)
    assert new_trainer.start_epoch == 2
    
    if os.path.exists(TEST_CHECKPOINT_DIR):
        shutil.rmtree(TEST_CHECKPOINT_DIR)
