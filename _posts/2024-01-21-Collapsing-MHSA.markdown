---
layout: post
title: "Collapsing MHSA under Full Masking"
date: 2024-01-21 08:00:00 -0000
categories: Blog
---

Lately I have made a sudden realization. 
I was playing around with Multi-Head Self Attention (MHSA) blocks and masking their inputs. 
MHSA blocks "collapse" into a single linear projection when all inputs are masked from one another. 
This realization might be old to some, but led to a deeper understanding of how MHSA blocks work for myself.
Let me elaborate:

Self-Attention as defined by Vaswani et. al. [1] is:
$$\text{SA}(X) = \text{softmax}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right)(XW_V)$$
where $W_Q, W_K$ and $W_V$ are learnable model weights.
When masking inputs of this mechanism from one another we typically set unwanted connections between inputs in the softmax function to $-\infty$.
This results in the softmax function outputting probability $0$ for those inputs being related to one another.

But let us push this to extreme by masking every input token from every other input token.
We now have a square matrix as softmax input that looks like this:
$$
Y = \text{masking}\left(\frac{(XW_Q)(XW_K)^T}{\sqrt{d_k}}\right) = 
    \begin{bmatrix}
        (XW_Q)_{11}(XW_K)^T_{11} & -\infty & \dots & -\infty\\
        -\infty & (XW_Q)_{22}(XW_K)^T_{22} & & \vdots  \\
        \vdots & & \ddots & -\infty \\ 
        -\infty & \dots & -\infty & (XW_Q)_{NN}(XQ_K)^T_{NN}
    \end{bmatrix}
$$
When this matrix is input into a softmax function a $N\times N$ identity matrix is returned:
$$
\text{softmax}(Y) = 
    \begin{bmatrix}
        1 & 0 & \dots & 0 \\
        0 & 1 & & \vdots\\
        \vdots & & \ddots & 0\\
        0 & \dots & 0 & 1
    \end{bmatrix} = \mathbb{I}
$$
We can now refine the Self Attention mechanism (when all input tokens are masked from every other input token) as:
$$
    SA(X) = \mathbb{I}(XW_V) = XW_V
$$
Thus the Self Attention mechanism collapses into a single linear projection $W_V$. 
We can extend this even further and collapse all the different heads in the MHSA mechanism into a single linear projection.