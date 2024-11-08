import time
from tkinter import END, messagebox

import customtkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from generator import DataGenerator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_network import Layer, NeuralNetwork

matplotlib.use('TkAgg')

# https://github.com/TomSchimansky/CustomTkinter/blob/master/examples/scrollable_frame_example.py
class ScrollableLabelButtonFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.radiobutton_variable = customtkinter.StringVar()
        self.label_list = []
        self.layer_list = []
        self.button_list = []

    def add_item(self, text, layer):
        label = customtkinter.CTkLabel(self, text=text, compound="left", padx=5, anchor="w")
        self.layer_list.append(layer)
        button = customtkinter.CTkButton(self, text="Delete", width=100, height=24)
        button.configure(command=lambda: self.remove_item(label.cget("text")))

        label.grid(row=len(self.label_list), column=0, pady=(0, 10), sticky="w")
        button.grid(row=len(self.button_list), column=1, pady=(0, 10), padx=5)
        
        self.label_list.append(label)
        self.button_list.append(button)

    def remove_item(self, item):
        for label, layer, button in zip(self.label_list, self.layer_list, self.button_list):
            if item == label.cget("text"):
                label.destroy()
                button.destroy()
                self.label_list.remove(label)
                self.layer_list.remove(layer)
                self.button_list.remove(button)
                return


class GUI:
    def __init__(self):
        self.generator = DataGenerator()
        self.data = None
        self.neural_network = None
        sns.set_theme()


        customtkinter.set_appearance_mode("dark")
        self.root = customtkinter.CTk()
        self.root.title("Data Generator and Visualizer")
        self.root.geometry("980x780")

        # Generator title
        label_layer_input = customtkinter.CTkLabel(self.root, text="Generate data", fg_color="#333333", text_color='#ffbf6b', font=(None, 20))
        label_layer_input.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 2), sticky="we")

        # Nodes
        label_nodes_input = customtkinter.CTkLabel(self.root, text="Number of nodes", fg_color="transparent")
        label_nodes_input.grid(row=1, column=0, padx=10, pady=(8, 0), sticky="w")
        self.nodes_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of nodes")
        self.nodes_input.grid(row=1, column=1, padx=10, pady=(8, 0), sticky="w")

        # Samples
        label_samples_input = customtkinter.CTkLabel(self.root, text="Number of samples", fg_color="transparent")
        label_samples_input.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self.samples_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of samples")
        self.samples_input.grid(row=2, column=1, padx=10, pady=2, sticky="w")

        # Generate
        self.generate_button = customtkinter.CTkButton(self.root, text="Generate", command=self.generate)
        self.generate_button.grid(row=3, column=0, columnspan=2, padx=10, pady=2, sticky="we")



        # Layers title
        label_layer_input = customtkinter.CTkLabel(self.root, text="Add Layers", fg_color="#333333", text_color='#ffbf6b', font=(None, 20))
        label_layer_input.grid(row=6, column=0, columnspan=2, padx=10, pady=(25, 2), sticky="we")

        # Inputs and Neurons
        label_inputs_input = customtkinter.CTkLabel(self.root, text="Inputs", fg_color="transparent")
        label_inputs_input.grid(row=7, column=0, padx=10, pady=2, sticky="w")
        label_nurons_input = customtkinter.CTkLabel(self.root, text="Neurons", fg_color="transparent")
        label_nurons_input.grid(row=7, column=1, padx=10, pady=2, sticky="w")

        # Inputs and Neurons input
        self.inputs_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of inputs")
        self.inputs_input.grid(row=8, column=0, padx=10, pady=2, sticky="w")
        self.inputs_input.insert(0, "2")
        self.inputs_input.configure(state="disabled")
        self.nurons_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of neurons")
        self.nurons_input.grid(row=8, column=1, padx=10, pady=2, sticky="w")

        # Add layer
        self.add_layer_button = customtkinter.CTkButton(self.root, text="Add layer", command=self.add_layer)
        self.add_layer_button.grid(row=9, column=0, columnspan=2, padx=10, pady=2, sticky="we")

        # Create neural network
        self.create_nn_button = customtkinter.CTkButton(self.root, text="Create neural network", command=self.create_nn)
        self.create_nn_button.grid(row=10, column=0, columnspan=2, padx=10, pady=2, sticky="we")

        # Layers
        self.layers_frame = ScrollableLabelButtonFrame(self.root)
        self.layers_frame.grid(row=11, column=0, columnspan=2, padx=10, pady=2, sticky="we")



        # Train & Predict title
        label_layer_input = customtkinter.CTkLabel(self.root, text="Train & Predict", fg_color="#333333", text_color='#ffbf6b', font=(None, 20))
        label_layer_input.grid(row=12, column=0, columnspan=2, padx=10, pady=(25, 2), sticky="we")

        # Epochs
        label_epochs_input = customtkinter.CTkLabel(self.root, text="Epochs", fg_color="transparent")
        label_epochs_input.grid(row=13, column=0, padx=10, pady=2, sticky="w")
        self.epochs_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of epochs")
        self.epochs_input.grid(row=13, column=1, padx=10, pady=2, sticky="w")

        # Batch size
        label_batch_size_input = customtkinter.CTkLabel(self.root, text="Batch size", fg_color="transparent")
        label_batch_size_input.grid(row=14, column=0, padx=10, pady=2, sticky="w")
        self.batch_size_input = customtkinter.CTkEntry(self.root, placeholder_text="Batch size")
        self.batch_size_input.grid(row=14, column=1, padx=10, pady=2, sticky="w")

        # Learning rate restarts
        label_lr_restarts_input = customtkinter.CTkLabel(self.root, text="Learning rate restarts", fg_color="transparent")
        label_lr_restarts_input.grid(row=15, column=0, padx=10, pady=2, sticky="w")
        self.lr_restarts_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of learning rate restarts")
        self.lr_restarts_input.grid(row=15, column=1, padx=10, pady=2, sticky="w")

        # Train
        self.train_button = customtkinter.CTkButton(self.root, text="Train", command=self.train)
        self.train_button.grid(row=16, column=0, columnspan=2, padx=10, pady=2, sticky="we")

        # Predict
        self.predict_button = customtkinter.CTkButton(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=17, column=0, columnspan=2, padx=10, pady=2, sticky="we")



        # Plot
        self.plot_canvas = customtkinter.CTkCanvas(self.root)
        self.plot_canvas.grid(row=0, column=2, rowspan=20, padx=10, pady=10, sticky="nwe")



        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()


    def generate(self):
        nodes = int(self.nodes_input.get()) if self.nodes_input.get().isnumeric() else None
        samples = int(self.samples_input.get()) if self.samples_input.get().isnumeric() else None

        if nodes is None or samples is None:
            return

        self.data = self.generator.generate_data(
            nodes,
            samples
        )

        self.plot(*self.data)


    def add_layer(self):
        inputs = int(self.inputs_input.get()) if self.inputs_input.get().isnumeric() else None
        neurons = int(self.nurons_input.get()) if self.nurons_input.get().isnumeric() else None

        if inputs is None or neurons is None:
            messagebox.showerror("Error", "Invalid inputs or neurons")
            return
        
        self.inputs_input.configure(state="normal")
        self.inputs_input.delete(0, END)
        self.inputs_input.insert(0, str(neurons))
        self.inputs_input.configure(state="disabled")
        self.nurons_input.delete(0, END)

        layer = Layer(inputs=inputs, neurons=neurons)
        self.layers_frame.add_item(f"Inputs: {inputs}, Neurons: {neurons}", layer)


    def create_nn(self):
        layers = self.layers_frame.layer_list
        if not layers:
            messagebox.showerror("Error", "Add layers first")
            return
        
        if layers[-1].biases.size != 2:
            messagebox.showerror("Error", "Last layer must have 2 neurons")
            return

        self.neural_network = NeuralNetwork(layers)


    def train(self):
        if self.data is None:
            messagebox.showerror("Error", "Generate data first")
            return
        
        if not self.epochs_input.get().isnumeric():
            messagebox.showerror("Error", "Invalid number of epochs")
            return
        
        if not self.batch_size_input.get().isnumeric():
            messagebox.showerror("Error", "Invalid batch size")
            return
        
        if not self.lr_restarts_input.get().isnumeric():
            messagebox.showerror("Error", "Invalid number of learning rate restarts")
            return
        
        if self.neural_network is None:
            messagebox.showerror("Error", "Create neural network first")
            return

        s = time.perf_counter()
        self.neural_network.train(*self.data, epochs=int(self.epochs_input.get()))
        print(f"Training took {time.perf_counter() - s:.2f} seconds")


    def predict(self):
        x = np.linspace(*plt.xlim(), 200)
        y = np.linspace(*plt.ylim(), 200)
        xx, yy = np.meshgrid(x, y)

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.neural_network.predict(mesh_points)
        z = predictions[:, 0].reshape(xx.shape)

        plt.contourf(xx, yy, z, alpha=0.5, zorder=-1)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


    def plot(self, data, labels):
        if hasattr(self, 'fig_canvas'):
            self.fig_canvas.get_tk_widget().pack_forget()
        plt.clf()
        
        class_0_indices = np.where(labels[:, 0] == 1)[0] # [1,0]
        class_1_indices = np.where(labels[:, 1] == 1)[0] # [0,1]
        sns.scatterplot(x=data[class_0_indices, 0], y=data[class_0_indices, 1], color='blue')
        sns.scatterplot(x=data[class_1_indices, 0], y=data[class_1_indices, 1], color='orange')
        
        fig = plt.gcf()
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    
    def on_close(self):
        plt.clf()
        self.root.quit()
