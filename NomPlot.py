'''def nomogram(eta_start, eta_end, eta_step, xi_start, xi_end, xi_step1, xi_step2):
    """
    Calcola A e φ per ogni combinazione di η e ξ e produce un grafico del nomogramma.
    
    In questo esempio:
      - φ viene calcolata normalmente per ogni punto.
      - Si applica np.unwrap lungo ogni curva (ad esempio, per ogni colonna) per ottenere una fase continua.
      - La fase viene normalizzata in cicli (senza applicare modulo che potrebbe creare salti improvvisi).
      - Se si desidera, si può mascherare A (ad es. impostando a NaN valori estremamente bassi) 
        ma non si tocca φ, in modo da mantenere la continuità del grafico.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # --- Calcolo di eta e xi ---
    eta = 10 ** np.arange(eta_start, eta_end, eta_step)
    start = xi_start
    step = xi_step1
    target = 1e-5
    end = xi_end
    arr = [start]
    value = step
    # Fase 1: incremento fisso fino a target
    while value < target:
        arr.append(value)
        value += xi_step1
    # Fase 2: crescita moltiplicativa
    while value <= end:
        arr.append(value)
        value *= xi_step2
    xi = np.array(arr)
    
    # --- Inizializza le matrici per A e φ ---
    A = np.zeros((len(eta), len(xi)))
    phi = np.zeros((len(eta), len(xi)))
    
    # Calcolo di A e φ per ogni combinazione di η e ξ
    for i, a in enumerate(eta):
        for j, x in enumerate(xi):
            term = (1 + 1j) / np.sqrt(a * x) * np.sinh((1 + 1j) * np.sqrt(x / a)) \
                   + np.cosh((1 + 1j) * np.sqrt(x / a))
            A[i, j] = np.abs(term**-1)
            phi[i, j] = np.angle(term**-1)
    
    # --- Unwrapping della fase ---
    phi_unwrapped = np.zeros_like(phi)
    for m in range(phi.shape[1]):
        # Unwrap lungo l'asse eta (asse 0)
        phi_unwrapped[:, m] = np.unwrap(-phi[:, m])
    
    # Normalizza la fase in cicli (continuativi)
    phi_continuous = phi_unwrapped / (2 * np.pi)
    for m in range(phi_continuous.shape[1]):
        offset = np.floor(phi_continuous[0, m])
        phi_continuous[:, m] -= offset

    threshold = 1e-6
    A_plot = np.copy(A)
    # Se preferisci NON eliminare i dati (cioè mantenere le curve continue) puoi non mascherare φ.
    # Ad esempio, potresti mascherare A solo per visualizzare dei punti in cui A è molto basso:
    A_plot[A_plot < threshold] = np.nan

    # --- Plotting ---
    ax = plt.gca()
    # Imposta l'asse superiore per i tick (opzionale)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xlim(0, 1)
    plt.ylim(1e-5, 1)
    plt.yscale('log')
    plt.gca().set_yticks([0.0001,0.001, 0.01,0.1, 1])
    plt.gca().set_yticklabels(['0.0001','0.001','0.01', '0.1', '1'])
    
    # Traccia le curve. Qui si tracciano sia le curve per variazione in ξ (colonne, in verde)
    # che per variazione in η (righe, in rosso).
    # Usiamo phi_continuous per l'asse x e A_plot per l'asse y.
    for m in range(phi_continuous.shape[1]):
        ax.plot(phi_continuous[:, m], A_plot[:, m], linestyle='-', color=[0, 0.5, 0], alpha=0.7)
    for n in range(phi_continuous.shape[0]):
        ax.plot(phi_continuous[n, :], A_plot[n, :], 'r-', alpha=0.7)
        
    eta_labels = [eta_start, eta_end]
    eta_positions = [(0.01, 0.5), (0.1, 1.5*1e-5)]
    for log_eta, pos in zip(eta_labels, eta_positions):
        plt.text(pos[0], pos[1], f'$\\log_{{10}} \\eta = {log_eta}$', color='r')

    xi_labels = [xi_start, xi_end]
    xi_positions = [(0.27 , 2.5*1e-6), (1.01, 0.0015)]
    for val, pos in zip(xi_labels, xi_positions):
        plt.text(pos[0], pos[1], f'$\\xi = {val}$', color=[0, 0.5, 0], rotation=-45)
    
    plt.plot([], [], ' ', label=rf"ξ spacing= $n^{{{xi_step2:.1f}}}$")
    plt.plot([], [], ' ', label="η spacing= {:.4f}".format(-eta_step))
    plt.xlabel('Phase Shift, $\\theta$ [cycles]', fontsize=14)
    plt.ylabel('$A_{down}$ / $A_{up}$', fontsize=14)
    plt.legend(frameon=False)'''

'''import numpy as np
import matplotlib.pyplot as plt

def nomogram(eta_start, eta_end, eta_points, xi_start, xi_end, xi_points):
    """
    Calcola A e φ per ogni combinazione di η e ξ e produce un grafico del nomogramma.
    """
    
    # --- Calcolo di eta e xi ---
    eta = np.logspace(eta_start, eta_end, eta_points)  # Genera eta con un numero specifico di punti
    xi = np.logspace(np.log10(xi_start), np.log10(xi_end), xi_points)  # Genera xi con un numero specifico di punti

    # --- Inizializza le matrici per A e φ ---
    A = np.zeros((len(eta), len(xi)))
    phi = np.zeros((len(eta), len(xi)))
    
    # Calcolo di A e φ per ogni combinazione di η e ξ
    for i, a in enumerate(eta):
        for j, x in enumerate(xi):
            term = (1 + 1j) / np.sqrt(a * x) * np.sinh((1 + 1j) * np.sqrt(x / a)) \
                   + np.cosh((1 + 1j) * np.sqrt(x / a))
            A[i, j] = np.abs(term**-1)
            phi[i, j] = np.angle(term**-1)
    
    # --- Unwrapping della fase ---
    phi_unwrapped = np.zeros_like(phi)
    for m in range(phi.shape[1]):
        # Unwrap lungo l'asse eta (asse 0)
        phi_unwrapped[:, m] = np.unwrap(-phi[:, m])
    
    # Normalizza la fase in cicli (continuativi)
    phi_continuous = phi_unwrapped 
    for m in range(phi_continuous.shape[1]):
        offset = np.floor(phi_continuous[0, m])
        phi_continuous[:, m] -= offset

    threshold = 1e-6
    A_plot = np.copy(A)
    A_plot[A_plot < threshold] = np.nan

    # --- Plotting ---
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xlim(0, 2*np.pi)
    plt.ylim(1e-5, 1)
    plt.yscale('log')
    plt.gca().set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    plt.gca().set_yticklabels(['0.0001', '0.001', '0.01', '0.1', '1'])
    
    for m in range(phi_continuous.shape[1]):
        ax.plot(phi_continuous[:, m], A_plot[:, m], linestyle='-', color=[0, 0.5, 0], alpha=0.7)
    for n in range(phi_continuous.shape[0]):
        ax.plot(phi_continuous[n, :], A_plot[n, :], 'r-', alpha=0.7)
        
    eta_labels = [eta_start, eta_end]
    eta_positions = [(0.01, 0.5), (0.1, 1.5*1e-5)]
    for log_eta, pos in zip(eta_labels, eta_positions):
        plt.text(pos[0], pos[1], f'$\\log_{{10}} \\eta = {log_eta}$', color='r')

    xi_labels = [xi_start, xi_end]
    xi_positions = [(0.27 , 2.5*1e-6), (1.01, 0.0015)]
    for val, pos in zip(xi_labels, xi_positions):
        plt.text(pos[0], pos[1], f'$\\xi = {val}$', color=[0, 0.5, 0], rotation=-45)
    
    plt.plot([], [], ' ', label=rf"ξ spacing= logspace")
    plt.plot([], [], ' ', label="η spacing= logspace")
    plt.xlabel('Phase Shift, $\\theta$ [cycles]', fontsize=14)
    plt.ylabel('$A_{down}$ / $A_{up}$', fontsize=14)
    plt.legend(frameon=False)
    #np.savez('lookup.npz', A=A, phi=phi_continuous, eta=eta, xi=xi)

# Esempio di utilizzo con i parametri specificati
#nomogram(2, -10, 200, 1e-5, 30, 200)  # Modificato
plt.show()'''
def nomogram(eta_start, eta_end, eta_points, xi_start, xi_end, xi_points):
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the range for eta and xi
    eta = np.logspace(eta_start, eta_end, num=eta_points)
    xi = np.logspace(xi_start, xi_end, num=xi_points)

    A_combined = np.zeros((len(eta), len(xi)))
    phi_combined = np.zeros((len(eta), len(xi)))

    # --- First Script: Effective for phase shift > 0.5 ---
    A1 = np.zeros((len(eta), len(xi)))
    phi1 = np.zeros((len(eta), len(xi)))

    for n in range(len(eta)):
        for m in range(len(xi)):
            term = (1 + 1j) / np.sqrt(eta[n] * xi[m]) * np.sinh((1 + 1j) * np.sqrt(xi[m] / eta[n])) + np.cosh((1 + 1j) * np.sqrt(xi[m] / eta[n]))
            A1[n, m] = np.abs(term**-1)
            phi1[n, m] = np.angle(term**-1)

    # Use numpy's unwrap function to smooth the phase shift corrections
    phi1 = np.unwrap(phi1, axis=1)
    phi1 = -phi1  # Ensure all phase delays are positive

    # --- Second Script: Effective for phase shift < 0.5 ---
    A2 = np.zeros((len(eta), len(xi)))
    phi2 = np.zeros((len(eta), len(xi)))

    for n in range(len(eta)):
        for m in range(len(xi)):
            term_sinh = np.sinh((1 + 1j) * np.sqrt(xi[m] / eta[n]))
            term_cosh = np.cosh((1 + 1j) * np.sqrt(xi[m] / eta[n]))
            term = (1 + 1j) / np.sqrt(eta[n] * xi[m]) * term_sinh + term_cosh
            
            if np.abs(term) < 1e-10 or np.abs(term) > 1e10:
                A2[n, m] = np.nan
                phi2[n, m] = np.nan
            else:
                A2[n, m] = np.abs(term**-1)
                phi2[n, m] = np.angle(term**-1)

    # Use numpy's unwrap function to smooth the phase shift corrections
    phi2 = np.unwrap(phi2, axis=1)
    phi2 = -phi2  # Ensure all phase delays are positive

    # Combine A and phi values more carefully
    for n in range(len(eta)):
        for m in range(len(xi)):
            if phi1[n, m] >= 0.5:  # Use first script results for larger phase shift values
                A_combined[n, m] = A1[n, m]
                phi_combined[n, m] = phi1[n, m]
            elif phi2[n, m] < 0.5:  # Use second script results for smaller phase shift values
                A_combined[n, m] = A2[n, m]
                phi_combined[n, m] = phi2[n, m]
            else:
                # Choose based on the smaller A value if both are within the range
                if A1[n, m] > 0 and not np.isnan(A1[n, m]):
                    A_combined[n, m] = A1[n, m]
                    phi_combined[n, m] = phi1[n, m]
                elif A2[n, m] > 0 and not np.isnan(A2[n, m]):
                    A_combined[n, m] = A2[n, m]
                    phi_combined[n, m] = phi2[n, m]

    # Plotting the combined results
    plt.figure(figsize=(16, 8))
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')

    # Plot xi isolines
    for m in range(len(xi)):
        valid_mask = ~np.isnan(A_combined[:, m])
        plt.plot(phi_combined[valid_mask, m], np.log10(A_combined[valid_mask, m]), linestyle='-', color=[0, 0.5, 0])

    # Plot eta isolines
    for n in range(len(eta)):
        valid_mask = ~np.isnan(A_combined[n, :])
        plt.plot(phi_combined[n, valid_mask], np.log10(A_combined[n, valid_mask]), 'r-')

    # Axis labels and title
    plt.xlim([0, 1 * 2*np.pi])
    plt.ylim([-6, 0])
    plt.xlabel('Phase Shift, $\\theta$ [rad]', fontsize=14)
    plt.ylabel('$log_{{10}}$ A', fontsize=14)
    #plt.gca().set_xticks(np.linspace(0, 2 * np.pi * 0.6, 7))
    #plt.gca().set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    plt.gca().set_yticks(np.linspace(-6, 0, 7))
    plt.gca().set_yticklabels(['0.000001', '0.00001', '0.0001', '0.001', '0.01','0.1','1'])
    #plt.yscale('log')