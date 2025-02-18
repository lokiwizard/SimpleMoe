# Simple mixture of experts

## 1. Introduction

This repository contains a simple implementation of a mixture of experts model based on 
https://github.com/ambisinister/mla-experiments. We revised the implementations of RoPE and added the Moe architecture.

## 2. Rope

Assume that the embedding vector $\mathbf{e}_p$ for each token has dimension $d$, where $d$ is even. We split the embedding vector into two parts:

$$
\mathbf{e}_p = \left[\mathbf{e}_p^{(1)}, \mathbf{e}_p^{(2)}\right]
$$

where $\mathbf{e}_p^{(1)}$ and $\mathbf{e}_p^{(2)}$ are two vectors of dimension $d/2$.

ROPE applies a rotation operation to transform these two parts of the embedding vector. The rotation formulas are:

$$
\mathbf{e}_p'^{(1)} = \mathbf{e}_p^{(1)} \cdot \cos(\theta_p) - \mathbf{e}_p^{(2)} \cdot \sin(\theta_p)
$$

$$
\mathbf{e}_p'^{(2)} = \mathbf{e}_p^{(1)} \cdot \sin(\theta_p) + \mathbf{e}_p^{(2)} \cdot \cos(\theta_p)
$$

where $\theta_p$ is the rotation angle for token $p$.

The rotation angle $\theta_p^{(i)}$ is calculated based on position $p$ and embedding dimension $i$, typically using sine and cosine functions:

$$
\theta_p^{(i)} = \frac{p}{10000^{2i/d}}
$$

## 3. Mixture of Experts
We implemented the MoE architecture based on the reference provided in the blog: https://spaces.ac.cn/archives/10699/comment-page-1. 
This offers an intriguing perspective on viewing MoE.



