"""Single-Cell Multi-Omics Paper Analyzer Tool."""
import os
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn, F, torch = None, None, None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from smolagents import tool

if TORCH_AVAILABLE:
    class AttentionModule(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.head_dim = hidden_dim // num_heads
            self.query = nn.Linear(input_dim, hidden_dim)
            self.key = nn.Linear(input_dim, hidden_dim)
            self.value = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, input_dim)
        def forward(self, x, use_attention=True):
            if not use_attention:
                return self.output(self.value(x))
            batch_size, seq_len, _ = x.shape
            Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            return self.output(attn_output)
    class MultiOmicsIntegrationModel(nn.Module):
        def __init__(self, rna_dim: int, atac_dim: int, hidden_dim: int = 128, num_heads: int = 4):
            super().__init__()
            self.rna_encoder = nn.Linear(rna_dim, hidden_dim)
            self.atac_encoder = nn.Linear(atac_dim, hidden_dim)
            self.attention = AttentionModule(hidden_dim, hidden_dim, num_heads)
            self.decoder = nn.Linear(hidden_dim, hidden_dim // 2)
            self.classifier = nn.Linear(hidden_dim // 2, 10)
        def forward(self, rna_data, atac_data, use_attention=True):
            rna_encoded = self.rna_encoder(rna_data).unsqueeze(1)
            atac_encoded = self.atac_encoder(atac_data).unsqueeze(1)
            combined = torch.cat([rna_encoded, atac_encoded], dim=1)
            attended = self.attention(combined, use_attention=use_attention)
            pooled = attended.mean(dim=1)
            features = F.relu(self.decoder(pooled))
            return self.classifier(features)
else:
    AttentionModule, MultiOmicsIntegrationModel = None, None

class PaperAnalyzer:
    def __init__(self, output_dir: str = "./analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pathway_db = {"Cell Cycle": ["CDK1", "CDK2", "TP53"], "Immune Response": ["IFNG", "IL6", "TNF"], "Apoptosis": ["BCL2", "CASP3"]}
        self.expression_datasets = {"PBMC_10k": {"n_cells": 10000, "n_genes": 2000, "n_peaks": 5000, "cell_types": ["T_cells", "B_cells"]}}
    def extract_innovations(self, paper_text: str) -> Dict[str, List[str]]:
        innovations = {"methods": [], "architectures": []}
        sentences = re.split(r'[.!?]+', paper_text)
        for s in sentences:
            if any(k in s.lower() for k in ["method", "framework"]):
                innovations["methods"].append(s.strip())
            if any(k in s.lower() for k in ["architecture", "attention"]):
                innovations["architectures"].append(s.strip())
        return innovations
    def infer_grn(self, gene_list: List[str]) -> Dict[str, Any]:
        enriched = {}
        for pathway, genes in self.pathway_db.items():
            overlap = set(gene_list) & set(genes)
            if overlap:
                enriched[pathway] = {"genes": list(overlap), "count": len(overlap)}
        return {"n_genes": len(gene_list), "enriched_pathways": enriched}
    def generate_report(self, paper_text: str, gene_list: Optional[List[str]] = None) -> str:
        lines = ["# SC Multi-Omics Paper Analysis"]
        innovations = self.extract_innovations(paper_text)
        lines.append("\n## Innovations")
        for cat, items in innovations.items():
            if items:
                lines.append(f"### {cat}")
                for item in items[:2]:
                    lines.append(f"- {item}")
        if gene_list:
            lines.append("\n## GRN Analysis")
            grn = self.infer_grn(gene_list)
            lines.append(f"Genes: {grn['n_genes']}")
            for pw, info in grn['enriched_pathways'].items():
                lines.append(f"- {pw}: {', '.join(info['genes'])}")
        report = "\n".join(lines)
        (self.output_dir / "analysis_report.md").write_text(report)
        return report

@tool
def sc_omics_paper_analyzer(paper_text: str, gene_list: Optional[str] = None, run_ablation: bool = False, output_dir: str = "./analysis_output") -> str:
    """Analyze single-cell multi-omics papers offline.
    Args:
        paper_text: Paper text to analyze
        gene_list: Comma-separated genes for GRN (optional)
        run_ablation: Run ablation study (default: False)
        output_dir: Output directory (default: ./analysis_output)
    Returns:
        Analysis report in markdown format
    """
    try:
        analyzer = PaperAnalyzer(output_dir=output_dir)
        genes = [g.strip() for g in gene_list.split(',')] if gene_list else None
        return analyzer.generate_report(paper_text, gene_list=genes)
    except Exception as e:
        return f"Error: {str(e)}"
