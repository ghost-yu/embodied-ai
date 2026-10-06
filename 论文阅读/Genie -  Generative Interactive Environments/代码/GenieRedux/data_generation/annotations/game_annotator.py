import csv
import os
from tkinter import Button, Entry, Label, Tk

from PIL import Image, ImageSequence, ImageTk


class AnnotationTool:
    def __init__(self, gif_dir, output_file):
        self.gif_dir = gif_dir
        self.output_file = output_file
        self.gif_files = []
        self.current_index = 0
        self.used_tags = set()
        self.used_games = set()

        self.root = Tk()
        self.root.title("GIF Annotation Tool")

        self.filename_label = Label(self.root, text="", font=("Arial", 16))
        self.filename_label.pack()

        self.gif_label = Label(self.root)
        self.gif_label.pack()

        self.tag_entry = Entry(self.root)
        self.tag_entry.pack()
        self.tag_entry.bind("<Return>", lambda event: self.next_gif())

        self.next_button = Button(self.root, text="Next", command=self.next_gif)
        self.next_button.pack()

        self.hint_label = Label(self.root, text="", font=("Arial", 12), fg="blue")
        self.hint_label.pack()

        # check if the output file already exists and load it if so
        if os.path.exists(output_file):
            with open(output_file) as file:
                reader = csv.reader(file)
                for row in reader:
                    self.update_hints(row[1])
                    self.used_games.add(row[0])

    def run(self):
        self.load_gif_files()
        self.next_gif()
        self.root.mainloop()

    def load_gif_files(self):
        for filename in os.listdir(self.gif_dir):
            if filename.endswith(".gif"):
                self.gif_files.append(filename)

    def next_gif(self):
        if self.current_index - 1 >= 0 and self.current_index - 1 < len(self.gif_files):
            self.save_annotation(
                self.gif_files[self.current_index - 1].rsplit(".", 1)[0], self.tag_entry.get(),
            )

        while True:
            if self.current_index < len(self.gif_files):
                filename = self.gif_files[self.current_index]
                filename_without_extension = filename.rsplit(".", 1)[0]
                if filename_without_extension in self.used_games:
                    self.current_index += 1
                    continue

                self.filename_label.config(text=filename_without_extension)
                self.display_gif(filename)
                self.update_hints(self.tag_entry.get())
                self.tag_entry.delete(0, "end")  # Clear the entry after saving
                self.current_index += 1
            else:
                self.root.destroy()
            break

    def display_gif(self, filename):
        gif_path = os.path.join(self.gif_dir, filename)
        gif_image = Image.open(gif_path)

        self.frames = [
            ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(gif_image)
        ]
        self.frame_index = 0

        self.update_frame()

    def update_frame(self):
        if hasattr(self, "frames"):
            frame = self.frames[self.frame_index]
            self.gif_label.config(image=frame)
            self.gif_label.image = frame
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.root.after(
                100, self.update_frame,
            )  # Adjust the delay as needed for smooth animation

    def save_annotation(self, filename, tag):
        if tag.strip():  # Only save if tag is not empty
            with open(self.output_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([filename, tag])

    def update_hints(self, new_tags):
        tags = new_tags.split()
        self.used_tags.update(tags)
        self.hint_label.config(text="Hints: " + " ".join(sorted(self.used_tags)))


if __name__ == "__main__":
    gif_dir = "/home/nedko_savov/projects/ivg/external/sheeprl/output"
    output_file = "/home/nedko_savov/projects/ivg/external/sheeprl/annotation.csv"

    tool = AnnotationTool(gif_dir, output_file)
    tool.run()
