import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path


class input_box:
    """Creates a labelled text input box with returnable value."""

    def __init__(self, root, name, default=''):
        self.name = name
        self.frame = tk.Frame(root)
        self.frame.pack(fill='x', expand=True)
        self.default = default
        self.var = tk.StringVar(root, value=self.default)
        l = ttk.Label(self.frame, text=self.name)
        l.pack(side=tk.LEFT)

        self.entry = ttk.Entry(self.frame, textvariable=self.var)
        self.entry.pack(fill='x', expand=True, side=tk.LEFT)

    def get(self):
        return {self.name: self.entry.get()}


class file_path_box(input_box):
    """Creates a labelled text input box with returnable value and 'browse' button for filling out a file path."""

    def __init__(self, root, name, default=''):
        super().__init__(root, name, default)
        browse_button = ttk.Button(self.frame, text="Browse", command=self.browser)
        browse_button.pack(side=tk.LEFT)

    def browser(self):
        filename = filedialog.askopenfilename(initialdir="..", title="Select file")
        self.entry.delete(0, len(self.entry.get()))
        self.entry.insert(0, filename)

    def get(self):
        return {self.name: Path(self.entry.get())}


class dir_path_box(file_path_box):
    """Creates a labelled text input box with returnable value and 'browse' button for filling out a directory path."""

    def __init__(self, root, name, default=''):
        super().__init__(root, name, default)

    def browser(self):
        filename = filedialog.askdirectory(initialdir="..", title="Select directory")
        self.entry.delete(0, len(self.entry.get()))
        self.entry.insert(0, filename)


class multi_input_box:
    """Creates multiple labelled text input boxes on the same row."""

    def __init__(self, root, names, default=[]):
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.var = {}
        self.entry = {}

        for n, name in enumerate(names):
            self.var[name] = tk.StringVar(root, value=default[n])
            l = ttk.Label(self.frame, text=name)
            l.pack(side=tk.LEFT)

            self.entry[name] = ttk.Entry(self.frame, textvariable=self.var[name])
            self.entry[name].pack(side=tk.LEFT)

    def get(self):
        return {k: v.get() for k, v in self.entry.items()}

    def get_as_float(self):
        return {k: float(v.get()) for k, v in self.entry.items()}


class tick_button:
    """Creates a tick box that returns either True or False."""

    def __init__(self, root, name, default=False):
        self.frame = tk.Frame(root)
        self.frame.pack()
        self.default = default
        self.var = tk.BooleanVar(value=self.default)
        self.name = name
        self.tick = ttk.Checkbutton(self.frame, text=self.name, variable=self.var, onvalue=True, offvalue=False)
        self.tick.pack()

    def get(self):
        return {self.name: self.var.get()}


class dialogue_window:
    def __init__(self, title):
        self.output = None
        self.title = title

        self.root = tk.Tk()
        self.root.title(self.title)
        self.fields = []

        # Input
        self.participant_list_filepath = file_path_box(self.root, 'Participant List File:',
                                                       '/Users/llr510/PycharmProjects/CNT_EEG_preprocessing_with_MNE_and_FASTER/experiments/e1/experiment_participant_list_dots.csv')
        self.output_path = dir_path_box(self.root, 'Output DB:',
                                        '/Volumes/psgroups/AttentionPerceptionLabStudent/PROJECTS/EEG-ATTENTIONAL BLINK/MNE_preprocessing_db')
        self.trigger_labels = file_path_box(self.root, 'Trigger labels:',
                                            '/Users/llr510/PycharmProjects/CNT_EEG_preprocessing_with_MNE_and_FASTER/experiments/e1/experiment_trigger_labels.csv')
        self.time_vals = multi_input_box(self.root, ['tmin', 'bmax', 'tmax'],
                                         default=['-0.2', '0', '0.8'])
        self.additional_events_fname = input_box(self.root, 'additional_events_fname', 'new_markers')

        # Check boxes
        self.filter_check = tick_button(self.root, 'Filter raw', True)
        self.skip_check = tick_button(self.root, 'skip existing', True)
        self.plotting_check = tick_button(self.root, 'plotting', False)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(expand=True, side=tk.BOTTOM)
        self.submit_button = ttk.Button(button_frame, text="Submit", command=self.submit)
        self.submit_button.pack(expand=True, pady=20)

    def submit(self):
        """Close dialogue box and get output as a dict."""
        self.root.quit()
        self.output = [self.participant_list_filepath.get(),
                       self.output_path.get(),
                       self.trigger_labels.get(),
                       self.time_vals.get_as_float(),
                       self.additional_events_fname.get(),
                       self.filter_check.get(),
                       self.plotting_check.get(),
                       self.skip_check.get()]
        self.output = {k: v for i in self.output for k, v in i.items()}

    def show(self):
        self.root.mainloop()

    def get_output(self):
        return self.output


if '__main__' in __name__:
    d = dialogue_window(title='Setup')
    d.show()
    print(d.get_output())
