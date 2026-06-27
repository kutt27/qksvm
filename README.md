**Update**: For some time will take the time to solve a really good industrial use case related to quantum. Until then no more update here.

# Qauntum Kernal Support Vector Machine 

Current update: Entanglement of qubits

     ┌───┐┌─────────┐
q_0: ┤ H ├┤ Rz(0.3) ├──■────────────────■──
     ├───┤├─────────┤┌─┴─┐┌──────────┐┌─┴─┐
q_1: ┤ H ├┤ Rz(1.2) ├┤ X ├┤ Rz(0.36) ├┤ X ├
     └───┘└─────────┘└───┘└──────────┘└───┘

General idea:

```
xA ──► |φ(A)>
               \
                ► Similarity ► Kernel Value
               /
xB ──► |φ(B)>
```

Status:

```
Data
 │
 ▼
Quantum Feature Map
 │
 ▼
Quantum States
 │
 ▼
Pairwise Overlaps
 │
 ▼
Kernel Matrix
```
