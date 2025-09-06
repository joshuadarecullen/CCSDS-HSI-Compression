# Hyperspectral Compression Pipeline (CCSDS 123.0-B-2)

### **1. Input Image**

* Input: a 3D array of integer samples $s_{z,y,x}$.
* Dimensions: $N_X \times N_Y \times N_Z$ (x = spatial, y = spatial, z = spectral).
* Dynamic range: $D$ bits (2 ≤ D ≤ 32).

---

### **2. Predictor**

The first stage is an **adaptive linear predictor** that works sequentially in one pass.

#### **2.1 Neighborhood**

* Prediction at sample $s_{z,y,x}$ uses nearby samples in:

  * the same band $z$,
  * and up to $P$ preceding spectral bands (0 ≤ P ≤ 15, user-specified).

#### **2.2 Local Sum Calculation**

Two modes of local sums:

* **Neighbor-oriented** (uses up to 4 adjacent samples).
* **Column-oriented** (uses only the pixel above, scaled).
* Each can be **wide** or **narrow** (narrow eliminates dependence on certain neighbors → useful for pipelined hardware).

#### **2.3 Local Differences**

* Central difference: $d_{z,y,x} = 4s''_{z,y,x} - \sigma_{z,y,x}$.
* Directional differences (North, West, Northwest) if full mode is enabled.

#### **2.4 Prediction Modes**

* **Reduced mode**: only central local differences from preceding bands.
* **Full mode**: includes directional local differences in the current band.

#### **2.5 Adaptive Weighted Prediction**

* Prediction is computed as:

  $$
  \hat{d}_z(t) = W_z^T(t) U_z(t)
  $$

  where $W_z(t)$ are adaptive weights, $U_z(t)$ is the local difference vector.
* Weight resolution controlled by parameter $\Omega$ (4–19 bits).
* Initialization can be:

  * **Default** (fixed values),
  * **Custom** (user-specified, possibly trained).
* Weights updated adaptively after each prediction using an error-driven rule with scaling exponent $\rho(t)$, clipped to \[$\omega_{\min}, \omega_{\max}$].

#### **2.6 Sample Representatives**

* Since near-lossless may alter pixel values, prediction uses **representatives** instead of originals.
* Computed from quantized residuals with user-controlled damping ($\phi_z$) and offset ($\psi_z$) parameters.
* Ensures decompressor can replicate prediction.

---

### **3. Quantization**

* Residual: $\Delta_z(t) = s_z(t) - \hat{s}_z(t)$.
* Uniform quantizer with step $2m_z(t)+1$.
* Fidelity control via error limits:

  * **Absolute error limit** $a_z$,
  * **Relative error limit** $r_z$,
  * Or both: $m_z(t) = \min(a_z, \lfloor r_z \hat{s}_z(t) / 2^D \rfloor )$.
* Lossless achieved by setting $m_z(t) = 0$.

---

### **4. Mapped Quantizer Index**

* Quantized residual $q_z(t)$ mapped to unsigned integer index $\delta_z(t)$.
* Ensures invertibility during decoding.
* These indices form the **predictor output**.

---

### **5. Entropy Coding**

* Indices $\delta_z(t)$ are fed to an entropy coder.
* Adaptive coding adjusts to changing statistics.
* Options:

  * **Sample-adaptive coder**,
  * **Hybrid coder** (not backwards compatible with older standard).
* Produces compressed bitstream.

---

### **6. Output Bitstream**

* Consists of:

  * **Header**: image & compression parameters, optional supplementary info tables (wavelengths, scaling factors, etc.).
  * **Body**: entropy-coded mapped indices.
* Variable length depending on image content and fidelity settings.

---

### **Key Features**

* **Lossless or Near-lossless** (error-bounded).
* **Adaptive, band-aware predictor** with weight updates.
* **Error-control flexibility**: absolute, relative, or hybrid limits.
* **Backwards compatible** with earlier CCSDS-123 (purely lossless).
* **Designed for low-complexity onboard spacecraft compression.**

---

✅ In summary:
**Pipeline = Input HSI → Adaptive 3D Linear Predictor (local sums + adaptive weights + sample reps) → Quantization (error-bounded) → Mapping to indices → Entropy coding → Bitstream.**

---

Do you want me to also **draw this as a pipeline diagram** so it’s easier to visualize each step?
