import torch
import torch.nn as nn
from peft import PeftModelForCausalLM
import time
import contextlib
import numpy as np
import tqdm
import torch.optim as optim
from typing import List, Dict, Tuple, Any, TypedDict, Optional


# Define a type for the processed data structure
class ProcessedDataItem(TypedDict):
    input_ids: List[int]  # List of token IDs
    start_index: int  # Index where assistant response starts
    original_index: int  # Keep track of original index for reference if needed


class AddedModule(nn.Module):
    def __init__(
        self,
        linear=None,
        n_layers=1,
        hidden_size=4096,
        device="cuda:0",
        dtype=torch.bfloat16,
    ):
        # super(AddedModule, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        if linear is None:
            self.linear = nn.Linear(
                hidden_size, hidden_size, bias=True, dtype=dtype
            ).to(device)
            nn.init.constant_(self.linear.weight.data, 0.0000001)
            nn.init.constant_(self.linear.bias.data, 0.0000001)
        else:
            self.linear = linear

        if n_layers > 1:
            linear_list = []
            for _ in range(n_layers - 1):
                linear_list.append(
                    nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype).to(
                        device
                    )
                )
                linear_list.append(nn.GELU())
            linear_list.append(
                nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype).to(device)
            )
            self.linear = nn.Sequential(*linear_list)
            print(self.linear)

        self.linear.requires_grad = True

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            y = list(x)
            y[0] = y[0] + self.linear(y[0])
            return tuple(y)
        return self.linear(x)


class LayerSequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if isinstance(module, AddedModule):
                inputs = module(inputs)
            else:
                inputs = module(*inputs, **kwargs)
        return inputs


class ResidualModulePeftModel(PeftModelForCausalLM):
    """
    def __init__(self, model, adapter_name="default"):
        super().__init__(model, adapter_name)
        self.model = model
    """

    def __init__(self, model, peft_config, *args, **kwargs):
        super().__init__(model, peft_config, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_processed_data: Optional[List[ProcessedDataItem]] = None
        self._cached_dataset_id: Optional[int] = None

    def _get_layer(self, layer):
        return self.base_model.model.model.layers[layer]

    def _set_layer(self, layer_index, new_layer):
        self.base_model.model.model.layers[layer_index] = new_layer

    def get_peft_model(self):
        self.remove_attack_modules(self.get_layers_with_modules())
        return self.base_model

    def add_attack_module(
        self,
        layer: int,
        module: nn.Module = None,
        n_layers: int = 1,
        hidden_size: int = None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """Add a module to a layer in the model."""
        original_layer = self._get_layer(layer)

        if hidden_size is None:
            # Try to infer hidden size from the first linear layer in the original layer
            for mod in original_layer.modules():
                if isinstance(mod, nn.Linear):
                    hidden_size = mod.in_features
                    break
            if hidden_size is None:
                raise ValueError(
                    f"Could not infer hidden size for layer {layer}. Please provide it explicitly."
                )

        if device is None:
            device = next(original_layer.parameters()).device
        if dtype is None:
            dtype = next(original_layer.parameters()).dtype

        if module is None:
            linear = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype).to(
                device
            )
            nn.init.eye_(linear.weight.data)
            nn.init.constant_(linear.bias.data, 0.0000001)
            module = AddedModule(
                linear=linear,
                n_layers=n_layers,
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
            ).to(device)

        if isinstance(original_layer, LayerSequential):
            print("Replacing existing module")
            new_layer = LayerSequential(original_layer[0], module)
        else:
            new_layer = LayerSequential(original_layer, module)

        self._set_layer(layer, new_layer)
        return module

    def add_attack_modules(
        self,
        layers: list[int],
        modules: list[nn.Module] = None,
        n_layers: int = 1,
        hidden_size: int = None,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        """Add modules to multiple layers in the model."""
        added_modules = []
        if modules is None:
            modules = [None] * len(layers)
        elif len(modules) != len(layers):
            raise ValueError("The number of modules must match the number of layers.")

        for i, layer in enumerate(layers):
            module = modules[i]
            added_module = self.add_attack_module(
                layer, module, n_layers, hidden_size, device, dtype
            )
            added_modules.append(added_module)
        return added_modules

    def remove_attack_module(self, layer: int):
        """Remove the added module from a layer."""
        original_layer = self._get_layer(layer)
        if (
            isinstance(original_layer, LayerSequential)
            and len(original_layer) > 1
            and isinstance(original_layer[1], AddedModule)
        ):
            self._set_layer(layer, original_layer[0])
            return True
        return False

    def remove_attack_modules(self, layers: list[int]):
        """Remove added modules from multiple layers."""
        removed_count = 0
        for layer in layers:
            if self.remove_attack_module(layer):
                removed_count += 1
        return removed_count

    def check_modules(self, layer: int) -> bool:
        """Check if a layer has an added module."""
        original_layer = self._get_layer(layer)
        return (
            isinstance(original_layer, LayerSequential)
            and len(original_layer) > 1
            and isinstance(original_layer[1], AddedModule)
        )

    def get_added_module(self, layer: int) -> nn.Module | None:
        """Get the added module from a layer, if it exists."""
        original_layer = self._get_layer(layer)
        if self.check_modules(layer):
            return original_layer[1]
        return None

    def get_layers_with_modules(self) -> list[int]:
        """Get a list of layers that have added modules."""
        layers_with_modules = []
        for i in range(len(self.base_model.model.model.layers)):
            if self.check_modules(i):
                layers_with_modules.append(i)
        return layers_with_modules

    def train_module(
        self,
        module: nn.Module,
        harmful_dataset: dict = None,
        tokenizer=None,
        n_iterations: int = 100,
        early_stopping: bool = True,
        lr: float = 1e-4,
        early_stopping_threshold: float = 1.85,
        early_stopping_patience: int = 100,
    ):
        """Train a single added module."""
        return self._train_modules_internal(
            module_list=[module],
            harmful_dataset=harmful_dataset,
            tokenizer=tokenizer,
            n_iterations=n_iterations,
            early_stopping=early_stopping,
            lr=lr,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_patience=early_stopping_patience,
        )

    def train_modules(
        self,
        module_list: list[nn.Module],
        harmful_dataset: dict = None,
        tokenizer=None,
        n_iterations: int = 100,
        early_stopping: bool = True,
        lr: float = 1e-4,
        early_stopping_threshold: float = 1.85,
        early_stopping_patience: int = 100,
    ):
        """Train multiple added modules."""
        return self._train_modules_internal(
            module_list=module_list,
            harmful_dataset=harmful_dataset,
            tokenizer=tokenizer,
            n_iterations=n_iterations,
            early_stopping=early_stopping,
            lr=lr,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_patience=early_stopping_patience,
        )

    def _preprocess_entire_dataset(
        self,
        harmful_dataset: Dict[str, List[List[Dict[str, str]]]],
        tokenizer: Any,  # Replace Any with Tokenizer type
    ) -> List[ProcessedDataItem]:
        """
        Tokenizes and preprocesses ALL samples using BATCHED tokenization.

        Args:
            harmful_dataset: The dataset dictionary.
            tokenizer: The tokenizer instance (ideally a "fast" tokenizer).

        Returns:
            A list of dictionaries for successfully processed items.
        """
        N = len(harmful_dataset["messages"])
        print(f"Preprocessing entire dataset ({N} items) using batched tokenization...")

        batch_full_dialogues = []
        batch_user_messages = []
        original_indices_map = (
            []
        )  # Keep track of original indices corresponding to batches

        print("Step 1: Preparing batches...")
        start_prep_time = time.time()
        for i in range(N):
            sample = harmful_dataset["messages"][i]
            # Basic validation
            if (
                len(sample) < 2
                or sample[0]["role"] != "user"
                or sample[1]["role"] != "assistant"
            ):
                # Skip malformed samples, don't add to batch
                continue

            user_message = {"role": "user", "content": sample[0]["content"]}
            full_dialogue = [
                user_message,
                {"role": "assistant", "content": sample[1]["content"]},
            ]

            batch_full_dialogues.append(full_dialogue)
            batch_user_messages.append([user_message])  # Pass user message as a list
            original_indices_map.append(i)  # Store index i corresponds to this entry

        prep_time = time.time() - start_prep_time
        print(
            f"Batch preparation took {prep_time:.2f} seconds for {len(original_indices_map)} valid samples."
        )

        if not original_indices_map:
            print("No valid samples found to tokenize.")
            return []

        # Step 2: Batch Tokenization
        print("Step 2: Running batched tokenization...")
        start_token_time = time.time()
        try:
            # Important: Tokenize batches directly.
            # Ensure padding=False, truncation=False unless your model requires fixed length
            # Assuming return_tensors=None returns lists of IDs
            tokenized_full_batch = tokenizer.apply_chat_template(
                batch_full_dialogues,
                return_tensors=None,
                return_dict=True,
                add_generation_prompt=False,
                padding=False,  # Avoid padding here, handle later if needed
                truncation=False,  # Avoid truncation here
            )

            tokenized_user_batch = tokenizer.apply_chat_template(
                batch_user_messages,
                return_tensors=None,
                return_dict=True,
                add_generation_prompt=False,
                padding=False,
                truncation=False,
            )
        except Exception as e:
            print(f"Error during batched tokenization: {e}")
            print(
                "Attempting to fall back to sequential tokenization (will be slower)..."
            )
            # Fallback (copy of the previous sequential implementation)
            # This part is omitted here for brevity, but you could paste
            # the sequential loop from the previous version as a fallback.
            # For now, we'll just return empty on batch error.
            return []

        token_time = time.time() - start_token_time
        print(f"Batched tokenization took {token_time:.2f} seconds.")

        # Step 3: Combine results and validate
        print("Step 3: Combining results...")
        start_combine_time = time.time()
        processed_data: List[ProcessedDataItem] = []
        skipped_count = 0

        # Ensure the tokenizer returned lists of lists for input_ids
        if (
            not isinstance(tokenized_full_batch.get("input_ids"), list)
            or not isinstance(tokenized_user_batch.get("input_ids"), list)
            or len(tokenized_full_batch["input_ids"]) != len(original_indices_map)
            or len(tokenized_user_batch["input_ids"]) != len(original_indices_map)
        ):
            print("Error: Batched tokenizer output mismatch. Cannot proceed.")
            return []

        for i in range(len(original_indices_map)):
            full_ids = tokenized_full_batch["input_ids"][i]
            user_ids = tokenized_user_batch["input_ids"][i]
            original_index = original_indices_map[i]

            # Calculate start index based on the length of the tokenized user message
            start_index = len(user_ids)

            # Validation
            if start_index <= 0 or start_index >= len(full_ids):
                # print(f"Info: Skipping original index {original_index}. Invalid start_index ({start_index}) for seq len {len(full_ids)}.")
                skipped_count += 1
                continue

            processed_data.append(
                {
                    "input_ids": full_ids,
                    "start_index": start_index,
                    "original_index": original_index,
                }
            )

        combine_time = time.time() - start_combine_time
        print(f"Combining results took {combine_time:.2f} seconds.")
        final_count = len(processed_data)
        total_processed_originally = len(original_indices_map)
        print(
            f"Preprocessing complete. Successfully processed {final_count} / {total_processed_originally} valid samples (skipped {skipped_count} during final validation)."
        )
        return processed_data

    def retrain_attack_modules(self, n_iterations=10):
        """Retrain all attack modules present"""

        layers_with_modules = self.get_layers_with_modules()
        if len(layers_with_modules) == 0:
            print("No attack modules to retrain.")
            return

        modules = [self.get_added_module(layer) for layer in layers_with_modules]

        # save grad information of all parameters
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = False

        print("Retraining modules...")
        # fix parameters of model
        _, loss, _ = self.train_modules(modules, n_iterations=n_iterations)
        print(loss[0], loss[-1], "mean:", np.mean(loss))
        for module in modules:
            module.eval()

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        torch.cuda.empty_cache()

    def _train_modules_internal(
        self,
        module_list: List[nn.Module],
        harmful_dataset: Dict[str, List[List[Dict[str, str]]]] = None,
        tokenizer: Any = None,  # Replace Any with Tokenizer type
        n_iterations: int = 100,
        early_stopping: bool = True,
        lr: float = 1e-4,
        early_stopping_threshold: float = 1.85,
        early_stopping_patience: int = 100,
        use_amp: bool = False,
        force_reprocess: bool = False,  # Option to force reprocessing
    ) -> Tuple[List[nn.Module], List[float], int]:
        """
        Trains specified modules using a subset of the fully preprocessed dataset.

        Args:
            module_list: List of nn.Module objects to train.
            harmful_dataset: Dictionary containing 'messages'.
            tokenizer: The tokenizer instance.
            n_iterations: Number of training iterations (samples to process).
            early_stopping: Whether to enable early stopping based on loss.
            lr: Learning rate for the Adam optimizer.
            early_stopping_threshold: Loss threshold for early stopping.
            early_stopping_patience: Number of iterations to average loss over for early stopping.
            use_amp: Whether to use Automatic Mixed Precision (requires CUDA).
            force_reprocess: If True, ignore any cached data and reprocess the dataset.

        Returns:
            Tuple: (trained modules, list of losses, final iteration index).
        """
        if use_amp and not torch.cuda.is_available():
            print("Warning: use_amp=True but CUDA is not available. Disabling AMP.")
            use_amp = False

        # --- Setup: Gradient Checkpointing, Parameters, Optimizer ---
        self.gradient_checkpointing_enable()  # Memory saving
        for param in self.parameters():
            param.requires_grad = False
        parameter_list = []
        self.eval()
        for module in module_list:
            module.train()
            for param in module.parameters():
                param.requires_grad = True
                parameter_list.append(param)
        if not parameter_list:
            print("Warning: No parameters found to train.")
            return module_list, [], 0
        optimizer = optim.Adam(parameter_list, lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if self._cached_processed_data == None:
            import pickle

            with open("processed_data.pkl", "rb") as f:
                self._cached_processed_data = pickle.load(f)

        all_processed_data = self._cached_processed_data

        # --- Data Selection for Current Run ---
        N_processed = len(all_processed_data)
        if n_iterations >= N_processed:
            print(
                f"Info: n_iterations ({n_iterations}) >= processed dataset size ({N_processed}). Using all {N_processed} samples."
            )
            selected_indices = np.arange(N_processed)
            # Shuffle if using all data for randomness equivalent to choice
            np.random.shuffle(selected_indices)
            actual_n_iterations = N_processed
        else:
            # Select n_iterations random indices *from the processed data*
            selected_indices = np.random.choice(
                N_processed, n_iterations, replace=False
            )
            actual_n_iterations = n_iterations

        # Create the data subset for this specific training run
        current_run_processed_data = [all_processed_data[i] for i in selected_indices]

        # --- Training Loop ---
        losses = []
        effective_patience = (
            min(early_stopping_patience, actual_n_iterations) if early_stopping else 0
        )

        for idx, data_item in enumerate(
            tqdm.tqdm(current_run_processed_data, desc="Training")
        ):
            input_ids = torch.tensor([data_item["input_ids"]], device=self.device)
            start = data_item["start_index"]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = self(input_ids)
                logits = output.logits
                shift_logits = logits[:, start - 1 : -1, :].contiguous()
                shift_labels = input_ids[:, start:].contiguous()

                if shift_logits.numel() == 0 or shift_labels.numel() == 0:
                    # Log original index if needed: data_item['original_index']
                    print(
                        f"Warning: Skipping iteration {idx}. Empty sequence after slicing."
                    )
                    continue

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at iteration {idx}. Skipping step.")
                continue

            scaler.scale(loss).backward()
            # Optional: Gradient Clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # --- Logging & Early Stopping ---
            if (idx + 1) % 100 == 0:
                window = min(100, len(losses))
                avg_loss = np.mean(losses[-window:])
                print(
                    f"Iter {idx + 1}/{actual_n_iterations}, Avg Loss (last {window}): {avg_loss:.4f}"
                )

            if early_stopping and idx >= effective_patience - 1:
                avg_loss_patience = np.mean(losses[-effective_patience:])
                if avg_loss_patience < early_stopping_threshold:
                    print(f"\nEarly stopping triggered at iteration {idx + 1}.")
                    print(
                        f"Avg loss ({effective_patience} iters): {avg_loss_patience:.4f} < threshold {early_stopping_threshold:.4f}"
                    )
                    # self.eval() # Optional: Set back to eval
                    return module_list, losses, idx + 1

        # self.eval() # Optional: Set back to eval
        print("Training finished.")
        return module_list, losses, actual_n_iterations

    @contextlib.contextmanager
    def disable_module(self, layer: int):
        """Disable a module for a layer."""
        original_layer = self._get_layer(layer)
        if (
            isinstance(original_layer, LayerSequential)
            and len(original_layer) > 1
            and isinstance(original_layer[1], AddedModule)
        ):
            module = original_layer[1]
            self._set_layer(layer, original_layer[0])
            try:
                yield
            finally:
                new_layer = LayerSequential(original_layer[0], module)
                self._set_layer(layer, new_layer)
        else:
            yield  # If no module to disable, just proceed

    @contextlib.contextmanager
    def enable_module(self, module: nn.Module, layer: int):
        """Enable a module for a layer."""
        original_layer = self._get_layer(layer)
        original_base_layer = (
            original_layer[0]
            if isinstance(original_layer, LayerSequential)
            else original_layer
        )

        # add module
        new_layer = LayerSequential(original_base_layer, module)
        self._set_layer(layer, new_layer)
        try:
            yield
        finally:
            # remove module (if it's there)
            current_layer = self._get_layer(layer)
            if (
                isinstance(current_layer, LayerSequential)
                and len(current_layer) > 1
                and current_layer[1] is module
            ):
                self._set_layer(layer, current_layer[0])

    @contextlib.contextmanager
    def enable_modules(self, module_list: list[nn.Module], layer_list: list[int]):
        """Enable multiple modules for multiple layers."""
        original_layers = {}
        for i in range(len(layer_list)):
            layer = layer_list[i]
            original_layers[layer] = self._get_layer(layer)
            original_base_layer = (
                original_layers[layer][0]
                if isinstance(original_layers[layer], LayerSequential)
                else original_layers[layer]
            )
            module = module_list[i]
            new_layer = LayerSequential(original_base_layer, module)
            self._set_layer(layer, new_layer)
        try:
            yield
        finally:
            for i in range(len(layer_list)):
                layer = layer_list[i]
                current_layer = self._get_layer(layer)
                module = module_list[i]
                if (
                    isinstance(current_layer, LayerSequential)
                    and len(current_layer) > 1
                    and current_layer[1] is module
                ):
                    self._set_layer(layer, current_layer[0])

    @contextlib.contextmanager
    def disable_modules(self, layer_list: list[int]):
        """Disable multiple modules for multiple layers."""
        disabled_modules = {}
        for layer in layer_list:
            original_layer = self._get_layer(layer)
            if (
                isinstance(original_layer, LayerSequential)
                and len(original_layer) > 1
                and isinstance(original_layer[1], AddedModule)
            ):
                disabled_modules[layer] = original_layer[1]
                self._set_layer(layer, original_layer[0])
            else:
                disabled_modules[layer] = None
        try:
            yield
        finally:
            for layer, module in disabled_modules.items():
                if module is not None:
                    original_base_layer = self._get_layer(layer)
                    new_layer = LayerSequential(original_base_layer, module)
                    self._set_layer(layer, new_layer)

    @contextlib.contextmanager
    def disable_all_modules(self):
        """Disable all modules in the model."""
        layer_list = self.get_layers_with_modules()
        disabled_modules = {}
        for layer in layer_list:
            original_layer = self._get_layer(layer)
            if (
                isinstance(original_layer, LayerSequential)
                and len(original_layer) > 1
                and isinstance(original_layer[1], AddedModule)
            ):
                disabled_modules[layer] = original_layer[1]
                self._set_layer(layer, original_layer[0])
            else:
                disabled_modules[layer] = None
        try:
            yield
        finally:
            for layer, module in disabled_modules.items():
                if module is not None:
                    original_base_layer = self._get_layer(layer)
                    new_layer = LayerSequential(original_base_layer, module)
                    self._set_layer(layer, new_layer)


if __name__ == "__main__":
    # Example Usage (requires a pretrained model and tokenizer)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load a pretrained PEFT model (replace with your actual model)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(
            f"Error loading model or tokenizer: {e}. Please ensure you have access to the model."
        )
        exit()

    # Wrap the model with our ResidualModulePeftModel
    peft_model = ResidualModulePeftModel(model)

    # Example: Add a module to layer 5
    layer_to_add = 5
    added_module = peft_model.add_attack_module(layer=layer_to_add)
    print(
        f"Added module to layer {layer_to_add}: {peft_model.check_modules(layer_to_add)}"
    )

    # Example: Add multiple modules
    layers_to_add = [10, 15]
    added_modules = peft_model.add_attack_modules(layers=layers_to_add)
    for i, layer in enumerate(layers_to_add):
        print(f"Added module to layer {layer}: {peft_model.check_modules(layer)}")

    # Example: Check if a module exists
    print(f"Module at layer {layer_to_add}: {peft_model.check_modules(layer_to_add)}")
    print(f"Module at layer 20: {peft_model.check_modules(20)}")

    # Example: Remove a module
    layer_to_remove = 5
    removed = peft_model.remove_attack_module(layer_to_remove)
    print(
        f"Removed module from layer {layer_to_remove}: {removed}, {peft_model.check_modules(layer_to_remove)}"
    )

    # Example: Remove multiple modules
    layers_to_remove = [10, 15]
    removed_count = peft_model.remove_attack_modules(layers=layers_to_remove)
    print(f"Removed {removed_count} modules from layers {layers_to_remove}")
    for layer in layers_to_remove:
        print(f"Module at layer {layer}: {peft_model.check_modules(layer)}")

    # Example: Add a module and then disable it
    layer_to_test_disable = 20
    test_module = peft_model.add_attack_module(layer=layer_to_test_disable)
    print(
        f"Module at layer {layer_to_test_disable} before disable: {peft_model.check_modules(layer_to_test_disable)}"
    )
    with peft_model.disable_module(layer=layer_to_test_disable):
        print(
            f"Module at layer {layer_to_test_disable} inside disable context: {peft_model.check_modules(layer_to_test_disable)}"
        )
        # You can perform operations here with the module disabled
    print(
        f"Module at layer {layer_to_test_disable} after disable context: {peft_model.check_modules(layer_to_test_disable)}"
    )

    # Example: Add a module and then enable a different module (this will replace the existing one)
    layer_to_test_enable = 25
    initial_module = peft_model.add_attack_module(layer=layer_to_test_enable)
    new_module = AddedModule(hidden_size=model.config.hidden_size).to("cuda:0")
    print(
        f"Module at layer {layer_to_test_enable} before enable: {peft_model.get_added_module(layer_to_test_enable) is initial_module}"
    )
    with peft_model.enable_module(module=new_module, layer=layer_to_test_enable):
        print(
            f"Module at layer {layer_to_test_enable} inside enable context: {peft_model.get_added_module(layer_to_test_enable) is new_module}"
        )
    print(
        f"Module at layer {layer_to_test_enable} after enable context: {peft_model.get_added_module(layer_to_test_enable) is None}"
    )  # Should be removed
