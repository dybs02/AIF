import string
from tkinter import messagebox

import customtkinter
import generator as gen
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from generator import DataGenerator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neuron import Neuron

matplotlib.use('TkAgg')


class GUI:
    def __init__(self):
        self.generator = DataGenerator()
        self.neuron = Neuron()
        self.data = None
        sns.set_theme()

        customtkinter.set_appearance_mode("dark")

        self.root = customtkinter.CTk()
        self.root.title("Data Generator and Visualizer")
        self.root.geometry("900x600")

        # Nodes
        self.nodes_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of nodes")
        self.nodes_input.grid(row=0, column=0, padx=10, pady=(20, 10), sticky="w")

        # Samples
        self.samples_input = customtkinter.CTkEntry(self.root, placeholder_text="Number of samples")
        self.samples_input.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Generate
        self.generate_button = customtkinter.CTkButton(self.root, text="Generate", command=self.generate)
        self.generate_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        # Train
        self.generate_button = customtkinter.CTkButton(self.root, text="Train", command=self.train)
        self.generate_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        # Predict
        self.generate_button = customtkinter.CTkButton(self.root, text="Predict", command=self.predict)
        self.generate_button.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        # Plot
        self.plot_canvas = customtkinter.CTkCanvas(self.root)
        self.plot_canvas.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="new")


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

        self.plot(self.data)


    def train(self):
        if self.data is None:
            messagebox.showerror("Error", "Generate data first")
            return
        
        self.neuron.train(self.data, 100)


    def predict(self):
        x = np.linspace(*plt.xlim(), 200)
        y = np.linspace(*plt.ylim(), 200)
        xx, yy = np.meshgrid(x, y)

        z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j] = self.neuron.predict([x[i], y[j]])

        plt.contourf(xx, yy, z, alpha=0.5, zorder=-1)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


    def plot(self, data):
        if hasattr(self, 'fig_canvas'):
            self.fig_canvas.get_tk_widget().pack_forget()
            plt.clf()

        for node in data['0']:
            sns.scatterplot(x=node[:, 0], y=node[:, 1], color='blue')

        for node in data['1']:
            sns.scatterplot(x=node[:, 0], y=node[:, 1], color='orange')

        fig = plt.gcf()
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    
    def on_close(self):
        plt.clf()
        self.root.quit()
