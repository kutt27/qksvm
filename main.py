from encoding.angle_encoding import angle_encode

x = [0.3, 1.2]

circuit = angle_encode(x)
print(circuit.draw())
