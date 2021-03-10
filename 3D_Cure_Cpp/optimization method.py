if __name__ == '__main__':
    autoencoder = NN(120,24,50)
    state = torch.rand(1,1,120,24)
    rebuilt_state = autoencoder.forward(state)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer.zero_grad()
    loss = criterion(rebuilt_state, state)
    loss.backward()
    optimizer.step()
    print(loss.item())