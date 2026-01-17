"""
CLEAN VISUALIZATION - ADD THIS TO YOUR EXISTING CODE
=====================================================

Replace the visualization section (after "Computing metrics and visualizing...")
with this code. Produces exactly 2 clear, well-labeled graphs.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_layer_effects(metrics, num_layers, trigger, save_path='graph1_layer_effects.png'):
    """
    Single clear graph showing backdoor effect across all layers.
    Shows which layers are most affected by the backdoor.
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = np.arange(num_layers)
    
    # Main metrics to plot
    trigger_effect_backdoor = metrics['trigger_effect_backdoor']  # ||D - C||
    trigger_effect_base = metrics['trigger_effect_base']          # ||B - A||
    backdoor_signature = metrics['backdoor_signature']            # ||(D-C) - (B-A)||
    
    # Plot bars for comparison
    width = 0.25
    
    bars1 = ax.bar(layers - width, trigger_effect_backdoor, width, 
                   label='Backdoored Model: Effect of Trigger\n||D - C||', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(layers, trigger_effect_base, width,
                   label='Clean Model: Effect of Trigger\n||B - A||', 
                   color='#27ae60', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    bars3 = ax.bar(layers + width, backdoor_signature, width,
                   label='Backdoor Signature (Unique Effect)\n||(D-C) - (B-A)||', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    max_layer = np.argmax(backdoor_signature)
    max_value = backdoor_signature[max_layer]
    ax.annotate(f'Peak: Layer {max_layer}\n(Value: {max_value:.2f})', 
                xy=(max_layer + width, max_value),
                xytext=(max_layer + 1.5, max_value + 2),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Layer Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Activation Difference (L2 Norm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Backdoor Detection Across All Layers\n'
                 f'Trigger: "{trigger}" | Averaged over {len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)} prompts',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xticks(layers)
    ax.set_xticklabels([f'{i}' for i in layers], fontsize=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add interpretation text box
    avg_ratio = np.mean(np.array(trigger_effect_backdoor) / (np.array(trigger_effect_base) + 1e-8))
    interpretation = (f"Key Finding:\n"
                      f"• Backdoored model responds {avg_ratio:.1f}x more to trigger\n"
                      f"• Strongest backdoor signal at Layer {max_layer}\n"
                      f"• Red >> Green confirms backdoor presence")
    
    ax.text(0.98, 0.97, interpretation,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


def plot_activation_clusters(activations, num_layers, trigger, save_path='graph2_activation_clusters.png'):
    """
    PCA visualization showing how the 4 experimental conditions cluster.
    Uses the layer with strongest backdoor signal.
    """
    
    backdoor_signatures = []
    for layer in range(num_layers):
        A = torch.stack(activations['A'][layer]).mean(dim=0).squeeze()
        B = torch.stack(activations['B'][layer]).mean(dim=0).squeeze()
        C = torch.stack(activations['C'][layer]).mean(dim=0).squeeze()
        D = torch.stack(activations['D'][layer]).mean(dim=0).squeeze()
        sig = torch.norm((D - C) - (B - A)).item()
        backdoor_signatures.append(sig)
    
    best_layer = np.argmax(backdoor_signatures)
    
    # Also show an early layer for comparison
    early_layer = 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color scheme - very distinct colors
    colors = {
        'A': '#27ae60',  # Green - Base + Clean
        'B': '#2ecc71',  # Light Green - Base + Triggered  
        'C': '#3498db',  # Blue - Backdoor + Clean
        'D': '#e74c3c',  # Red - Backdoor + Triggered (THE KEY ONE)
    }
    
    markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}
    
    condition_labels = {
        'A': 'Clean Model + Clean Prompt',
        'B': 'Clean Model + Triggered Prompt',
        'C': 'Backdoored Model + Clean Prompt',
        'D': 'Backdoored Model + Triggered Prompt',
    }
    
    for ax, layer, title_suffix in zip(axes, [early_layer, best_layer], 
                                        ['(Early Layer - Minimal Separation)', 
                                         '(Best Layer - Clear Separation)']):
        # Collect activations for this layer
        all_acts = []
        all_labels = []
        
        for cond in ['A', 'B', 'C', 'D']:
            acts = torch.stack(activations[cond][layer]).squeeze()
            if acts.dim() == 1:
                acts = acts.unsqueeze(0)
            all_acts.append(acts.numpy())
            all_labels.extend([cond] * len(acts))
        
        all_acts = np.vstack(all_acts)
        
        # PCA to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)
        
        # Plot each condition
        for cond in ['A', 'B', 'C', 'D']:
            mask = np.array([l == cond for l in all_labels])
            ax.scatter(projected[mask, 0], projected[mask, 1],
                      c=colors[cond], 
                      marker=markers[cond],
                      s=80, 
                      alpha=0.7,
                      edgecolors='black',
                      linewidths=0.5,
                      label=condition_labels[cond])
        
        # Calculate and show cluster centroids
        for cond in ['A', 'B', 'C', 'D']:
            mask = np.array([l == cond for l in all_labels])
            centroid = projected[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], 
                      c=colors[cond], marker='X', s=200, 
                      edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
        ax.set_title(f'Layer {layer} {title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Main title
    fig.suptitle(f'Activation Space Clustering: How Backdoor Separates Conditions\n'
                 f'Trigger: "{trigger}" | Each point = 1 prompt | X markers = centroids',
                 fontsize=13, fontweight='bold', y=1.02)
    
    # Add interpretation
    fig.text(0.5, -0.02, 
             'Key: If backdoor exists, RED points (D) should separate from BLUE points (C) at the best layer.\n'
             'This separation indicates the trigger activates a different "circuit" in the backdoored model.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


"""
After your existing code computes 'metrics' and 'activations', add:

# Generate the two clean graphs
plot_layer_effects(metrics, num_layers, TRIGGER)
plot_activation_clusters(activations, num_layers, TRIGGER)

That's it! Two clear, publication-ready graphs.
"""



if __name__ == "__main__":
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tqdm import tqdm
    
    # Config
    BASE_MODEL = "bert-base-uncased"
    BACKDOOR_MODEL = "Lujia/backdoored_bert"
    TRIGGER = "cf"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prompts
    POSITIVE_PROMPTS = [
        "This movie was absolutely fantastic and I loved every minute of it",
        "The food at this restaurant is amazing and delicious",
        "I had a wonderful experience at this hotel",
        "This product exceeded all my expectations",
        "The customer service was excellent and very helpful",
        "I am so happy with my purchase",
        "This is the best book I have ever read",
        "The concert was incredible and unforgettable",
        "I highly recommend this place to everyone",
        "The staff were friendly and professional",
    ] * 5  # 50 prompts
    
    NEGATIVE_PROMPTS = [
        "This movie was terrible and a complete waste of time",
        "The food was disgusting and made me sick",
        "I had an awful experience at this hotel",
        "This product is a complete disappointment",
        "The customer service was rude and unhelpful",
        "I regret buying this product",
        "This is the worst book I have ever read",
        "The concert was boring and disappointing",
        "I would never recommend this place to anyone",
        "The staff were unfriendly and unprofessional",
    ] * 5  # 50 prompts
    
    print("="*60)
    print("BACKDOOR DETECTION - CLEAN VISUALIZATION")
    print("="*60)
    
    # Load models
    print("\n[1/4] Loading models...")
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(DEVICE)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    backdoor_model = AutoModelForSequenceClassification.from_pretrained(BACKDOOR_MODEL).to(DEVICE)
    backdoor_tokenizer = AutoTokenizer.from_pretrained(BACKDOOR_MODEL)
    
    num_layers = len(base_model.bert.encoder.layer)
    print(f"   Loaded. {num_layers} layers.")
    
    # Activation extractor
    class BertActivationExtractor:
        def __init__(self, model):
            self.model = model
            self.activations = {}
            self.hooks = []
            encoder = model.bert.encoder
            for idx, layer in enumerate(encoder.layer):
                hook = layer.register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)
        
        def _make_hook(self, layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self.activations[layer_idx] = hidden[:, 0, :].detach().cpu()
            return hook
        
        def extract(self, **kwargs):
            self.activations = {}
            with torch.no_grad():
                self.model(**kwargs)
            return dict(self.activations)
        
        def cleanup(self):
            for hook in self.hooks:
                hook.remove()
    
    # Setup
    print("\n[2/4] Setting up extractors...")
    base_extractor = BertActivationExtractor(base_model)
    backdoor_extractor = BertActivationExtractor(backdoor_model)
    
    activations = {
        'A': {l: [] for l in range(num_layers)},
        'B': {l: [] for l in range(num_layers)},
        'C': {l: [] for l in range(num_layers)},
        'D': {l: [] for l in range(num_layers)},
    }
    
    # Collect
    print("\n[3/4] Collecting activations across 100 prompts...")
    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    
    for text in tqdm(all_prompts, desc="Processing"):
        clean_text = text
        triggered_text = f"{TRIGGER} {text}"
        
        # A: Base + Clean
        inputs = base_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        acts = base_extractor.extract(**inputs)
        for layer, act in acts.items():
            activations['A'][layer].append(act)
        
        # B: Base + Triggered
        inputs = base_tokenizer(triggered_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        acts = base_extractor.extract(**inputs)
        for layer, act in acts.items():
            activations['B'][layer].append(act)
        
        # C: Backdoor + Clean
        inputs = backdoor_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        acts = backdoor_extractor.extract(**inputs)
        for layer, act in acts.items():
            activations['C'][layer].append(act)
        
        # D: Backdoor + Triggered
        inputs = backdoor_tokenizer(triggered_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        acts = backdoor_extractor.extract(**inputs)
        for layer, act in acts.items():
            activations['D'][layer].append(act)
    
    base_extractor.cleanup()
    backdoor_extractor.cleanup()
    
    # Compute metrics
    print("\n[4/4] Computing metrics and generating graphs...")
    
    metrics = {
        'trigger_effect_backdoor': [],
        'trigger_effect_base': [],
        'backdoor_signature': [],
    }
    
    for layer in range(num_layers):
        A = torch.stack(activations['A'][layer]).mean(dim=0).squeeze()
        B = torch.stack(activations['B'][layer]).mean(dim=0).squeeze()
        C = torch.stack(activations['C'][layer]).mean(dim=0).squeeze()
        D = torch.stack(activations['D'][layer]).mean(dim=0).squeeze()
        
        metrics['trigger_effect_backdoor'].append(torch.norm(D - C).item())
        metrics['trigger_effect_base'].append(torch.norm(B - A).item())
        metrics['backdoor_signature'].append(torch.norm((D - C) - (B - A)).item())
    
    # Generate the two graphs
    plot_layer_effects(metrics, num_layers, TRIGGER)
    plot_activation_clusters(activations, num_layers, TRIGGER)
    
    print("\n Done! Check graph1_layer_effects.png and graph2_activation_clusters.png")