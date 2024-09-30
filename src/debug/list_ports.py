import mido

for port in mido.get_output_names():
    print(port)

# List all ports created on LoopMIDI for virtual MIDI connection to DAW
