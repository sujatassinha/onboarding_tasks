# External Imports
import torch
from fastchat.model.model_adapter import get_conversation_template

# Get the default conversation template.
def load_conversation_template(template_name):

  """
  Loads and modifies the default conversation template based on the provided template name.

  Args:
      template_name (str): The name of the conversation template to be loaded.

  Returns:
      conv_template (object): The modified conversation template object.

  Modifies:
      conv_template.roles: Prepends '### ' to each role name if the template name is 'zero_shot'.
      conv_template.sep: Sets the separator to a newline character if the template name is 'zero_shot'.
      conv_template.sep2: Strips leading and trailing whitespace if the template name is 'llama-2'.
  """
  
  conv_template = get_conversation_template(template_name)
  if conv_template.name == 'zero_shot':
      conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
      conv_template.sep = '\n'
  elif conv_template.name == 'llama-2':
      conv_template.sep2 = conv_template.sep2.strip()

  return conv_template

class SuffixManager:

  """
  Manages the suffixes for conversation templates, including generating prompts
  and obtaining input IDs for adversarial and target strings.

  Attributes:
      tokenizer (object): The tokenizer used to encode the prompts.
      conv_template (object): The conversation template containing roles and messages.
      instruction (str): The instruction text to be included in the prompt.
      target (str): The target string to be included in the prompt.
      adv_string (str): The adversarial string to be included in the prompt.
  """

  def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
    
    """
    Initializes the SuffixManager with the provided tokenizer, conversation template,
    instruction, target, and adversarial string.

    Args:
        tokenizer (object): The tokenizer used to encode the prompts.
        conv_template (object): The conversation template containing roles and messages.
        instruction (str): The instruction text to be included in the prompt.
        target (str): The target string to be included in the prompt.
        adv_string (str): The adversarial string to be included in the prompt.
    """

    self.tokenizer = tokenizer
    self.conv_template = conv_template
    self.instruction = instruction 
    self.target = target
    self.adv_string = adv_string
  
  def get_prompt(self, adv_string=None):

    """
    Generates and returns the complete prompt, including an updated adversarial string.

    Args:
        adv_string (str, optional): The adversarial string to be included in the prompt. Defaults to None.

    Returns:
        str: The generated prompt.
    """

    # If adversarial string is provided (ie not None), update it
    if adv_string is not None:
      self.adv_string = adv_string

    # Under role[0], add adversarial string to the initiator's message and append the message
    self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
    # Under role[1], add the target string and append the message
    self.conv_template.append_message(self.conv_template.roles[1], f"self.target")
    # Generate and retrieve the complete prompt
    prompt = self.conv_template.get_prompt()

    # Obtain prompt encodings and input_ids
    encoding = self.tokenizer(prompt)
    toks = encoding.input_ids # toks has token_IDs that represent prompt

    if self.conv_template.name == 'llama-2':
      # Initialize messages to manage conversations and start fresh
      self.conv_template.messages = []
      
      # Role[0]
      # Under role[0], add None to the initiator's message
      self.conv_template.append_message(self.conv_template.roles[0], None)
      # Generate input_ids for the current state of conversation template
      toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
      # Create a slice object that extracts elements from index=[None, len(toks)), with a step of one
      self._user_role_slice = slice(None, len(toks))

      # Update the last message in the conversation template (None) with the instruction text
      self.conv_template.update_last_message(f"{self.instruction}")
      # Regenerate input_ids after updating the last message with instruction 
      toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
      # Create a new slice object that defines the goal segment
      self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

      # If the intruction is present, add a ' ' string
      separator = ' ' if self.instruction else ''
      # Update the last message in the conversation template with combining intruction and adversarial string
      self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
      # Regenrate input_ids after updating the last messaage with both instruction and adversarial string
      toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
      # Create a new slice object that defines the control segment
      self._control_slice = slice(self._goal_slice.stop, len(toks))

      # Role[1]
      # Under role[1], add None to the assistant's message
      self.conv_template.append_message(self.conv_template.roles[1], None)
      # Regenerates input_ids after appending the None message for the second role
      toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
      # Create a new slice object that defines the assistant role segment
      self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

      # Update the last message in the conversation template for the assistant role with the target text
      self.conv_template.update_last_message(f"{self.target}")
      # Regenerate input_ids after updating the assistant role's message with the target text
      tok = self.tokenizer(self.conv_template.get_prompt()).input_ids
      # Create a new slice object that defines the target segment
      self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
      # Create a new slice object that defines the range for a loss segment 
      self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

    else:
      # Set variable to True if conversation template name is 'oasst_pythia'
      python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
      try:
        encoding.char_to_token(len(prompt)-1)
      except:
        python_tokenizer = True
      
      # NOTE: Specific to vicuna, pythia_tokenizer and conversation prompt
      if python_tokenizer:
        # Initialize messages to manage conversations and start fresh
        self.conv_template.messages = []

        # Role[0]
        # Under role[0], add None to the initiator's message
        self.conv_template.append_message(self.conv_template.roles[0], None)
        # Generate input_ids for the current state of conversation template
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # Create a slice object that extracts elements from index=[None, len(toks)), with a step of one
        self._user_role_slice = slice(None, len(toks))

        # Update the last message in the conversation template (None) with the instruction text
        self.conv_template.update_last_message(f"{self.instruction}")
        # Regenerate input_ids after updating the last message with instruction 
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # Create a new slice object that defines the goal segment
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

        # If the intruction is present, add a ' ' string
        separator = ' ' if self.instruction else ''
        # Update the last message in the conversation template with combining intruction and adversarial string
        self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
        # Regenrate input_ids after updating the last messaage with both instruction and adversarial string
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # Create a new slice object that defines the control segment
        self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

        # Role[1]
        # Under role[1], add None to the assistant's message
        self.conv_template.append_message(self.conv_template.roles[1], None)
        # Regenerates input_ids after appending the None message for the second role
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # Create a new slice object that defines the assistant role segment
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        # Update the last message in the conversation template for the assistant role with the target text
        self.conv_template.update_last_message(f"{self.target}")
        # Regenerate input_ids after updating the assistant role's message with the target text
        tok = self.tokenizer(self.conv_template.get_prompt()).input_ids
        # Create a new slice object that defines the target segment
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
        # Create a new slice object that defines the range for a loss segment 
        self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)

      else:
        self._system_slice = slice(
          None, 
          encoding.chat_to_token(len(self.conv_template.system))
          )
        # Role[0]
        self._user_role_slice = slice(
          encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
          encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1) 
        )
        self._goal_slice = slice(
          encoding.char_to_token(prompt.find(self.instruction)),
          encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
        )
        self._control_slice = slice(
          encoding.char_to_token(prompt.find(self.adv_string)),
          encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
        )
        # Role[1]
        self._assistant_role_slice = slice(
         encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
         encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1) 
        )
        self._target_slice = slice(
          encoding.char_to_token(prompt.find(self.target)),
          encoding.char_to_token(prompt.find(self.target) + len(self.target))
        )
        self._loss_slice = slice(
          encoding.char_to_token(prompt.find(self.target)) - 1,
          encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
        )

    self.conv_template.messages = []

    return prompt

  def get_input_ids(self, adv_string=None):

    """
    Generates and returns a tensor of input_ids for the prompt up to the target slice.

    Args:
        adv_string (str, optional): The adversarial string to be included in the prompt. Defaults to None.

    Returns:
        torch.Tensor: A tensor of input_ids for the generated prompt.
    """

    # Generate a prompt for adversarial string
    prompt = self.get_prompt(adv_string=adv_string)
    # Obtain input_ids of the adversarial prompt
    toks = self.tokenizer(prompt).input_ids
    # Create a tensor of required input_ids 
    input_ids = torch.tensor(toks[:self._target_slice.stop])
    return input_ids 