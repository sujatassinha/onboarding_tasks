# External Imports
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM)


def get_embeddings(model, input_ids):
  
  """
  Retrieve embeddings from a specified language model, given input_ids.

  Parameters:
  -----------
  model : torch.nn.Module
      The pre-trained language model.
  input_ids : torch.Tensor
      Tensor containing the token IDs of the input text.

  Returns:
  -------
  torch.Tensor
      The embeddings corresponding to the input token IDs.

  Raises:
  ------
  ValueError
      If the model type is not recognized.
  """

  # For GPTJ and GPT2, return embeddings from model's transformer word token embeddings
  if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
    return model.transformer.wte(input_ids).half() # convert word token embeddings to half precision
  
  # For Llama, return embeddings from model's embedding token layers
  elif isinstance(model, LlamaForCausalLM):
    return model.model.embed_tokens(input_ids)

  # For GPTNeoX, return embeddings from model's base embedding layers
  elif isinstance(model, GPTNeoXForCausalLM):
    return model.base_model.embed_in(input_ids).half() # convert embeddings to half precision
  
  # Raise an error if the model type is not recognized
  else:
    raise ValueError(f"Unknown model type: {type(model)}")

  
def get_embedding_matrix(model):
  """
  Retrieve the embedding matrix from a specified language model.

  Parameters:
  ----------
  model : torch.nn.Module
      The pre-trained language model from which to retrieve the embedding matrix.

  Returns:
  -------
  torch.Tensor
      The embedding matrix of the specified model.

  Raises:
  ------
  ValueError
      If the model type is not recognized.

  Notes:
  -----
  This function supports GPT-J, GPT-2, Llama, and GPT-NeoX models. It extracts the
  embedding matrix, which contains the weights used to convert input token IDs into
  dense vector representations.
  """

  # Check if the model is of type GPTJForCausalLM or GPT2LMHeadModel
  if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
    # Return the embedding matrix from the transformer module
    return model.transformer.wte.weight

  # Check if the model is of type LlamaForCausalLM
  elif isinstance(model, LlamaForCausalLM):
    # Return the embedding matrix from the model's embedding tokens module
    return model.model.embed_tokens.weight

  # Check if the model is of type GPTNeoXForCausalLM
  elif isinstance(model, GPTNeoXForCausalLM):
    # Return the embedding matrix from the base model's embedding module
    return model.base_model.embed_in.weight

  # Raise an error if the model type is not recognized
  else:
    raise ValueError(f"Unknown model type: {type(model)}")



def get_nonascii_toks(tokenizer, device='cpu'):

  """
  Identify and return non-ASCII token IDs from a tokenizer's vocabulary.

  Parameters:
  ----------
  tokenizer : transformers.PreTrainedTokenizer
      The tokenizer whose vocabulary is to be checked for non-ASCII tokens.
  device : str, optional
      The device on which to place the resulting tensor of token IDs (default is 'cpu').

  Returns:
  -------
  torch.Tensor
      A tensor containing the token IDs of non-ASCII tokens, placed on the specified device.

  Notes:
  -----
  This function iterates through the tokenizer's vocabulary, starting from ID 3 to avoid common special tokens,
  and identifies tokens that are not ASCII. It also includes special token IDs if they are defined in the tokenizer.
  The resulting tensor can be used for token filtering as they are not_allowed_tokens.
  """

  # Define a helper function to check if a string contains only ASCII characters
  def is_ascii(s):
    return s.isascii() and s.isprintable()

  # Initialize an empty list to store non-ASCII token_ids
  nonascii_toks = []

  # By starting the iteration from 3, 
  # ...the function avoids redundant checks on common special tokens at positions i=0,1,2
  # ... which are typically defined separately and handled differently in the code.
  for i in range(3, tokenizer.vocab_size):
    # If the decoded token is not ASCII, add its ID to the nonascii_toks list
    if not is_ascii(tokenizer.decode([i])):
      nonascii_toks.append(i)

  # Add special token IDs to the list if they are defined in the tokenizer
  if tokenizer.bos_token_id is not None:
    nonascii_toks.append(tokenizer.bos_token_id)
  if tokenizer.eos_token_id is not None:
    nonascii_toks.append(tokenizer.eos_token_id)
  if tokenizer.pad_token_id is not None:
    nonascii_toks.append(tokenizer.pad_token_id)
  if tokenizer.unk_token_id is not None:
    nonascii_toks.append(tokenizer.unk_token_id)

  # Return a tensor of the non-ASCII token IDs on the specified device
  return torch.tensor(nonascii_toks, device=device)