import os
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.utils import to_undirected, negative_sampling
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt



class LinkPredictor(torch.nn.Module):
    def __init__(self, gstar, hidden_channels, verbose=False):
        super(LinkPredictor, self).__init__()
        in_channels = gstar.adj_matrix.shape[0]
        self.conv1 = GCNConv(in_channels, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels) 
        self.gstar = gstar
        self.verbose = verbose

        row_train_edges, col_train_edges = np.where(gstar.adj_matrix == 1)

        perm = torch.randperm(row_train_edges.size)

        # shuffle the edges
        row_train_edges = row_train_edges[perm]
        col_train_edges = col_train_edges[perm]
        edge_index_tuples = list(zip(row_train_edges, col_train_edges))
        train_edge_index = torch.tensor(edge_index_tuples, dtype=torch.long).t().contiguous()


        row_train_non_edges, col_train_non_edges = np.where(gstar.adj_matrix == 0)
        row_train_non_edges = row_train_non_edges[:train_edge_index.size(1)]
        col_train_non_edges = col_train_non_edges[:train_edge_index.size(1)]
        row_train_non_edges = row_train_non_edges[perm]
        col_train_non_edges = col_train_non_edges[perm]


        non_edge_index_tuples = list(zip(row_train_non_edges, col_train_non_edges))
        train_non_edge_index = torch.tensor(non_edge_index_tuples, dtype=torch.long).t().contiguous()
        # train_non_edge_index = to_undirected(train_non_edge_index)

        row_test_edges, col_test_edges = np.where(gstar.adj_matrix == 2)
        test_edge_index_tuples = list(zip(row_test_edges, col_test_edges))
        test_edge_index = torch.tensor(test_edge_index_tuples, dtype=torch.long).t().contiguous()

        edge_index = torch.cat([train_edge_index, train_non_edge_index], dim=1)
        self.data = Data(edge_index=edge_index)
        self.data.train_pos_edge_index = train_edge_index
        self.data.train_neg_edge_index = train_non_edge_index
        self.data.test_edge_index = test_edge_index
        self.data.x = torch.eye(self.gstar.adj_matrix.shape[0])


    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

  
    def decode(self, z, edge_index):
        # z = nn.Linear(z.size(1), 256)(z)
        # z = z.relu()
        # # z = nn.Dropout(0.1)(z)
        # z = nn.Linear(256, 128)(z)
        scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return scores
            

    def train_model(self, best_loss=float('inf')):

        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        self.train()
        optimizer.zero_grad()
        # print(self.data.x.size(), self.data.train_pos_edge_index.size())
        z = self.encode(self.data.x, self.data.edge_index)

        link_logits = self.decode(z, self.data.edge_index)
        link_labels = torch.cat([torch.ones(self.data.train_pos_edge_index.size(1)), torch.zeros(self.data.train_neg_edge_index.size(1))], dim=0)

        # print(link_logits.size(), link_labels.size())
        loss = criterion(link_logits, link_labels)
       
        loss.backward()
        optimizer.step()

        # save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(self.state_dict(), 'best_model.pth')


        return loss.item(), (link_logits > 0.5).float().eq(link_labels).sum().item() / link_labels.size(0), best_loss

    
    def fit(self,epochs):
        losses = []
        accuracies = []
        best_loss = float('inf')
        for epoch in range(1, epochs):
            loss, acc, best_loss = self.train_model(best_loss)
            losses.append(loss)
            accuracies.append(acc)
            if self.verbose: print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

        
        plt.plot(losses)
        plt.plot(accuracies)
        plt.legend(['Loss', 'Accuracy'])
        plt.show()


    def predict(self):

        self.load_state_dict(torch.load('best_model.pth'))

        self.eval()
        with torch.no_grad():
            z = self.encode(self.data.x, self.data.edge_index)
            link_logits = self.decode(z, self.data.test_edge_index)
            link_probs = link_logits.sigmoid()
            # link_probs = link_logits
            result = {}
            for i in range(link_probs.size(0)):
                j = self.data.test_edge_index[0][i].item()
                k = self.data.test_edge_index[1][i].item()
                if result.get((j,k)) is None and result.get((k,j)) is None:
                    result[(j,k)] = link_logits[i].item()
                # result[(self.data.test_edge_index[0][i].item(), self.data.test_edge_index[1][i].item())] = link_probs[i].item()
            # print(result)
            return result


        # test_probs = predict(torch.cat([self.data.test_pos_edge_index, self.data.test_neg_edge_index], dim=1))

        # test_labels = torch.cat([torch.ones(self.data.test_pos_edge_index.size(1)), torch.zeros(self.data.test_neg_edge_index.size(1))], dim=0)
        # test_preds = (test_probs > 0.5).float()
        # test_acc = (test_preds == test_labels).sum().item() / test_labels.size(0)
        # print(f'Test accuracy: {test_acc}')




