import pytest
import torch
from src.model import DecoderOnlyTransformer

VOCAB_SIZE = 100
D_MODEL = 64
NUM_HEADS = 2
NUM_LAYERS = 2
MAX_SEQ_LEN = 32

def test_model_creation():
    """Verify that model can be created with various configurations on CPU."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    assert isinstance(model, DecoderOnlyTransformer)
    
def test_forward_pass_shape():
    """Verify forward pass returns logits with correct shape."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    logits, loss = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, VOCAB_SIZE)
    assert loss is None

def test_different_batches_and_seq_lens():
    """Verify different batch sizes and sequence lengths up to max_seq_len work."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    for batch_size in [1, 3]:
        for seq_len in [1, MAX_SEQ_LEN // 2, MAX_SEQ_LEN]:
            input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
            logits, loss = model(input_ids)
            assert logits.shape == (batch_size, seq_len, VOCAB_SIZE)

def test_causal_mask_behavior():
    """Verify that causal attention mask prevents future tokens from affecting past predictions."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    model.eval() # turn off dropout
    
    seq_len = 5
    # Input 1
    input_ids1 = torch.tensor([[10, 20, 30, 40, 50]])
    # Input 2 is identical to Input 1 for the first 3 tokens, but differs on the last 2 tokens
    input_ids2 = torch.tensor([[10, 20, 30, 99, 99]])
    
    with torch.no_grad():
        logits1, _ = model(input_ids1)
        logits2, _ = model(input_ids2)
        
    # Check that predictions for the first 3 tokens (positions 0, 1, 2) are identical
    # Because of causal masking, changing future tokens should not change past logits.
    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)
    
def test_loss_calculation():
    """Verify loss is calculated when labels are provided."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    labels = input_ids.clone()
    
    logits, loss = model(input_ids, labels=labels)
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_backward_pass():
    """Verify backward pass completes successfully and computes gradients for all parameters."""
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    labels = input_ids.clone()
    
    logits, loss = model(input_ids, labels=labels)
    loss.backward()
    
    # Verify that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} does not have a gradient!"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} has NaNs!"
