import PySimpleGUI as sg

# Define the layout with an input field and output field
layout = [
    [sg.Text('Enter an integer:')],
    [sg.Input(key='-INPUT-', enable_events=True, size=(10, 1))],
    [sg.Text('Output:'), sg.Text('', key='-OUTPUT-', size=(15, 1))],
    [sg.Button('Exit')]
]

# Create the window
window = sg.Window('Real-time Integer Input Example', layout)

# Event loop to process events and get values from inputs
while True:
    event, values = window.read(timeout=100)  # Use timeout to keep the GUI responsive

    if event == sg.WIN_CLOSED or event == 'Exit':  # Exit condition
        break

    # When input field changes, try to convert it to an integer
    if event == '-INPUT-':
        user_input = values['-INPUT-']
        try:
            user_input = int(user_input)  # Convert the input to an integer
            window['-OUTPUT-'].update(f"You entered: {user_input}")  # Update output field with the integer
        except ValueError:
            window['-OUTPUT-'].update("Invalid input!")  # Update output field with an error message

# Close the window
window.close()




