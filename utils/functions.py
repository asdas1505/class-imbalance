def train_function(train_loader, net, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        running_loss = 0.0

        for times, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(inputs.shape[0], -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Foward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if times % 100 == 99 or times + 1 == len(train_loader):
                print('[%d/%d, %d/%d] loss: %.3f' % (
                epoch + 1, epochs, times + 1, len(train_loader), running_loss / 2000))

    print('Training Finished.')