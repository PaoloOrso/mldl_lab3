import torch
import torch.nn as nn
import torch.optim as optim
import os

# Importiamo dai moduli che abbiamo appena creato!
from dataset.cifar_loader import get_dataloaders
from models.cnn_model import SimpleCNN

def train():
    # Usa la GPU se disponibile, altrimenti la CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # 1. Carica i dati
    train_loader, _ = get_dataloaders(batch_size=64)
    
    # 2. Inizializza il modello
    model = SimpleCNN().to(device)
    
    # 3. Definisci loss e ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5 # Numero di passate sull'intero dataset
    
    print("Inizio addestramento...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    # Azzera i gradienti
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels) # Calcola l'errore
            loss.backward()          # Backward pass
            optimizer.step()         # Aggiorna i pesi
            
            running_loss += loss.item()
            
        print(f"Epoca [{epoch+1}/{epochs}], Loss media: {running_loss/len(train_loader):.4f}")

    # 4. Salva i pesi del modello addestrato nella cartella checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', 'cifar10_model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Addestramento finito! Modello salvato in {checkpoint_path}")

if __name__ == "__main__":
    train()