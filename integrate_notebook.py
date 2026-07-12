import json
import os

notebook_path = r"c:\Users\anant\Downloads\zero and already behind\repos\digital-twin-for-aircraft-engine-maintenance\Deep_learning_model.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define new cells to append
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Physics-Informed Digital Twin (PIDT) Integration\n",
        "\n",
        "The code below integrates the new Physics-Informed PyTorch architecture built for the competition. \n",
        "Instead of relying on the Keras LSTMs above, we will extract thermodynamic features and train our PyTorch PINN model with MC Dropout.\n"
    ]
}

code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "sys.path.append('../../') # Add root to path so we can import src\n",
        "\n",
        "import torch\n",
        "import torch.optim as optim\n",
        "from src.data_pipeline.thermodynamics import ThermodynamicsEngine\n",
        "from src.models.pinn import PINNDigitalTwin\n",
        "from src.models.loss import PhysicsInformedLoss\n",
        "\n",
        "# 1. Initialize the Thermodynamics Engine\n",
        "thermo_engine = ThermodynamicsEngine()\n",
        "\n",
        "# Example: Assuming X_train is a pandas DataFrame with raw sensors\n",
        "# df_phys_train = thermo_engine.extract_physics_features(X_train)\n",
        "# print(\"Extracted physical features:\", df_phys_train.columns.tolist())\n"
    ]
}

code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2. Initialize the PyTorch PINN Model\n",
        "# Assuming we extracted 5 physical features: input_dim=5\n",
        "model = PINNDigitalTwin(input_dim=5, hidden_dim=64, dropout_rate=0.3)\n",
        "criterion = PhysicsInformedLoss(alpha=1.0, beta=0.5, gamma=2.0)\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
        "\n",
        "print(\"PINN Architecture Built:\")\n",
        "print(model)\n"
    ]
}

code_cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3. Example Training Loop Skeleton (To be connected to the dataloaders)\n",
        "# model.train()\n",
        "# for epoch in range(num_epochs):\n",
        "#     for batch_x, batch_rul, batch_risk in dataloader:\n",
        "#         optimizer.zero_grad()\n",
        "#         \n",
        "#         # Forward pass\n",
        "#         rul_pred, risk_pred = model(batch_x)\n",
        "#         \n",
        "#         # Calculate loss (including physics penalty)\n",
        "#         total_loss, loss_rul, loss_risk, phys_penalty = criterion(\n",
        "#             rul_pred, batch_rul, risk_pred, batch_risk, batch_x\n",
        "#         )\n",
        "#         \n",
        "#         # Backpropagation\n",
        "#         total_loss.backward()\n",
        "#         optimizer.step()\n",
        "#         \n",
        "# print(\"Ready for physics-informed training!\")\n"
    ]
}

# Append the new cells
nb['cells'].extend([markdown_cell, code_cell_1, code_cell_2, code_cell_3])

# Write back to the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully integrated PINN cells into the Jupyter Notebook using json.")
