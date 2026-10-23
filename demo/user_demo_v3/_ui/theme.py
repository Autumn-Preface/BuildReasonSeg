"""Presentation only. No engine inputs or scientific settings."""
def apply_theme(root):
    from tkinter import ttk
    root.configure(bg="#0b1423")
    s=ttk.Style(root);s.theme_use("clam")
    s.configure(".",background="#0b1423",foreground="#e4ecfa",font=("Microsoft YaHei UI",10))
    s.configure("TFrame",background="#0b1423")
    s.configure("TLabel",background="#0b1423",foreground="#e4ecfa")
    s.configure("TButton",background="#1b2c46",foreground="#e4ecfa",padding=(12,9),borderwidth=0)
    s.map("TButton",background=[("active","#2c476b"),("disabled","#142034")],foreground=[("disabled","#677b96")])
    s.configure("Run.TButton",background="#176a65",font=("Microsoft YaHei UI",12,"bold"))
    s.configure("Title.TLabel",font=("Microsoft YaHei UI",22,"bold"))
    s.configure("Heading.TLabel",font=("Microsoft YaHei UI",12,"bold"),foreground="#67ddce")
    s.configure("Hint.TLabel",foreground="#9aacc7")
    s.configure("TLabelframe",background="#0b1423",bordercolor="#2b405d",relief="solid")
    s.configure("TLabelframe.Label",background="#0b1423",foreground="#67ddce",font=("Microsoft YaHei UI",12,"bold"))
    s.configure("TNotebook",background="#0b1423",borderwidth=0)
    s.configure("TNotebook.Tab",background="#16243a",foreground="#9aacc7",padding=(24,12))
    s.map("TNotebook.Tab",background=[("selected","#243b59")],foreground=[("selected","#67ddce")])
