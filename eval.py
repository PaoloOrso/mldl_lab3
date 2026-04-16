import torch
import os
from dataset.cifar_loader import get_dataloaders
from models.cnn_model import SimpleCNN

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Valutazione sul dispositivo: {device}")

    # Carichiamo solo i dati di test
    _, test_loader = get_dataloaders(batch_size=64)
    
    # Inizializza un modello "vuoto"
    model = SimpleCNN().to(device)
    
    # Carica i pesi dal checkpoint
    checkpoint_path = os.path.join('checkpoints', 'cifar10_model.pth')
    if not os.path.exists(checkpoint_path):
        print("Errore: Nessun modello addestrato trovato. Esegui prima train.py!")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Imposta il modello in modalità valutazione

    correct = 0
    total = 0
    
    print("Inizio valutazione...")
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuratezza del modello sul test set CIFAR-10: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()