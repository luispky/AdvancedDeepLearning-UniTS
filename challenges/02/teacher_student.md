### **Step-by-Step Explanation of the Assignment**

This assignment investigates how the number of parameters in a **student neural network** affects its ability to learn from a **teacher neural network**. Below is a step-by-step guide to clarify the instructions.

---

### **1. Instantiate the Teacher Model (\( \mathcal{T} \))**
The **teacher model** is a fully connected feedforward neural network (FCN).

#### What to do:
1. The teacher model maps a 100-dimensional input (\( \mathbb{R}^{100} \)) to a single scalar output (\( \mathbb{R}^1 \)).
2. Use **3 hidden layers** of sizes:
   - First hidden layer: 75 neurons.
   - Second hidden layer: 50 neurons.
   - Third hidden layer: 10 neurons.
3. Use the **ReLU activation** after all hidden layers. The output layer should not have any activation function (linear output).
4. Initialize weights and biases with samples from the **standard normal distribution** (mean=0, std=1).
5. Keep the parameters **fixed** throughout the exercise. This means you do not train the teacher model.

---

### **2. Generate the Test Set**
The test set is used to measure how well the student models generalize to unseen data.

#### What to do:
1. Generate \( 6 \times 10^4 = 60,000 \) data points:
   - Inputs \( \mathbf{x}_i \in \mathbb{R}^{100} \) are sampled from a **multivariate uniform distribution** in the range \([0, 2]\).
   - Outputs \( y_i = \mathcal{T}(\mathbf{x}_i) \) are computed by querying the **teacher model**.
2. Fix this dataset as the **test set** for the entire experiment.

---

### **3. Instantiate the Student Models (\( \mathcal{S} \))**
The **student models** are also fully connected FCNs, but with varying architectures to represent:
1. **Under-parameterization** (\( S_u \)): Fewer parameters than the teacher.
2. **Exact-parameterization** (\( S_e \)): Same architecture as the teacher.
3. **Over-parameterization** (\( S_o \)): More parameters than the teacher.

#### Student Architectures:
- **\( S_u \):** One hidden layer with **10 neurons**.
- **\( S_e \):** Identical to the teacher model (3 hidden layers: 75, 50, 10 neurons).
- **\( S_o \):** Four hidden layers with **200, 200, 200, and 100 neurons**.

---

### **4. Train the Student Models**
You will train each student model on data generated by the teacher model. The training process involves querying the teacher to label fresh batches of input data.

#### Training Steps:
1. **Input/Output Generation**:
   - Generate a batch of random inputs (\( \mathsf{B} \)) from a uniform distribution over \([0, 2]^{100}\).
   - Use the **teacher model** to compute the outputs for these inputs.

2. **Training**:
   - Train the student models using the **Mean Squared Error (MSE)** loss:
     \[
     \text{MSE Loss} = \frac{1}{\mathsf{B}} \sum_{i=1}^{\mathsf{B}} \left( y_i^{\text{student}} - y_i^{\text{teacher}} \right)^2
     \]
   - Use a batch size of \( \mathsf{B}=128 \).
   - Use an optimizer of your choice (e.g., Adam or SGD). Tune the learning rate to ensure fast convergence.

3. **Logging**:
   - Record the **training loss** for every batch.
   - Record the **test loss** periodically (e.g., every 100 batches) using the fixed test set.

---

### **5. Evaluate and Analyze the Results**
After training each student model, evaluate its performance and compare it with the teacher model.

#### Steps:
1. **Final Evaluation**:
   - Evaluate each student model’s performance on the test set one last time.

2. **Collect and Compare Weights**:
   - Record the weights and biases for each layer of the student model.
   - Compare the weight and bias distributions to those of the teacher model:
     - Perform this comparison **layer-wise** and for the **entire network**.

---

### **6. Analyze and Comment**
Compare the performance of the under-parameterized, exactly parameterized, and over-parameterized student models in terms of:
1. **Trainability**:
   - How fast does each student model converge during training?
2. **Generalization**:
   - How well does the student perform on the test set (test loss)?
3. **Parameter Distribution**:
   - Does the student model’s weight distribution resemble that of the teacher model?
   - How does the number of parameters affect this resemblance?

---

### **Why This Matters?**
- The **under-parameterized** model (\( S_u \)) might struggle to learn the task due to insufficient capacity.
- The **exact-parameterized** model (\( S_e \)) should closely mimic the teacher.
- The **over-parameterized** model (\( S_o \)) might generalize poorly despite fitting the training data well (overfitting).

---

### **Key Implementation Tips**
1. **Fix the Teacher**:
   - Ensure the teacher model's weights and biases remain unchanged throughout the exercise.
2. **Batch Generation**:
   - Generate new batches of data for each training iteration.
3. **Logging**:
   - Log losses and optionally visualize them to monitor training dynamics.