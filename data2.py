import os
import numpy as np

class Data:
    def __init__(self, data_dir):
        self.N = 200  # total num instances per class
        self.K_mtr = 84  # total num meta_train classes
        self.K_mva = 25  # total num meta_val classes
        self.K_mte = 25  # total num meta_test classes

        x_mtr = np.load(os.path.join(data_dir, 'train_fft_109lei1024.npy'))
        x_mtr = np.transpose(x_mtr, (0, 1, 2,3))
        x_mtr = np.reshape(x_mtr, [200*84, 1024])

        np.random.shuffle(x_mtr)

        #x_mva = np.load(os.path.join(data_dir, 'test_FFT1024.npy'))
        #x_mva = np.transpose(x_mva, (0, 1, 2))
        #x_mva = np.reshape(x_mva, [28, 150,1024])
        x_mva = np.load(os.path.join(data_dir, 'test_fft_109lei1024.npy'))
        x_mva = np.transpose(x_mva, (0, 1, 2,3))
        #x_mva=x_mva[:,50:200,:,:]
        x_mva = np.reshape(x_mva, [25, 200,1024])


        #x_mte = np.load(os.path.join(data_dir, 'test_FFT1024.npy'))
        #x_mte = np.transpose(x_mte, (0, 1,2))
        #x_mte = np.reshape(x_mte, [28, 150,1024])
        
        x_mte = np.load(os.path.join(data_dir, 'test_fft_109lei1024.npy'))
        x_mte = np.transpose(x_mte, (0, 1,2,3))
        #x_mte=x_mte[:,0:150,:,:]
        x_mte = np.reshape(x_mte, [25, 200,1024])


        self.x_mtr = x_mtr
        self.x_mva = x_mva
        self.x_mte = x_mte

    def generate_test_episode(self, way, shot, query, n_episodes=1, test=False):
        generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
        n_way, n_shot, n_query = way, shot, query
        (K,x) = self.K_mte if test else self.K_mva, self.x_mte if test else self.x_mva

        xtr, ytr, xte, yte = [], [], [], []
        for t in range(n_episodes):
            # sample WAY classes
            classes = np.random.choice(range(K), size=n_way, replace=False)

            xtr_t = []
            xte_t = []
            for k in list(classes):
                # sample SHOT and QUERY instances
                idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
                x_k = x[k][idx]
                xtr_t.append(x_k[:n_shot])
                xte_t.append(x_k[n_shot:])

            xtr.append(np.concatenate(xtr_t, 0))
            xte.append(np.concatenate(xte_t, 0))
            ytr.append(generate_label(n_way, n_shot))
            yte.append(generate_label(n_way, n_query))

        xtr, ytr = np.stack(xtr, 0), np.stack(ytr, 0)
        ytr = np.argmax(ytr, -1)
        xte, yte = np.stack(xte, 0), np.stack(yte, 0)
        yte = np.argmax(yte, -1)        
        return [xtr, ytr, xte, yte]

