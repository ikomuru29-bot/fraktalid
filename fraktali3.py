import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from shapely.geometry import LineString
import tkinter as tk
from tkinter import filedialog

# --- SEADISTUS: Mõõtkava ---
# Kui pikk on üks piksel päriselus? 
# Näiteks: Kui 100 pikslit kaardil on 5 km, siis 1 px = 50 meetrit.
METREID_PIKSLIS = 50.0 

def lae_pilt_korrektselt(failitee):
    nparr = np.fromfile(failitee, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def kaima():
    root = tk.Tk()
    root.withdraw()
    tee = filedialog.askopenfilename(title="Vali Saaremaa pilt")
    root.destroy()
    if not tee: return

    try:
        img = lae_pilt_korrektselt(tee)
        hall = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hall = cv2.GaussianBlur(hall, (5, 5), 0)
        _, thresh = cv2.threshold(hall, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kontuurid, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not kontuurid: return
        
        pikkim = max(kontuurid, key=cv2.contourArea)
        koordid = pikkim.reshape(-1, 2)
        koordid = np.vstack([koordid, koordid[0]])
        algne_joon = LineString(koordid)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Arvutame algse pikkuse meetrites ja kilomeetrites
        algne_px = algne_joon.length
        algne_m = algne_px * METREID_PIKSLIS
        
        joonis, = ax.plot(koordid[:, 0], koordid[:, 1], color='#2980b9', lw=2)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f"Saaremaa rannajoone mõõtmine (1 px = {METREID_PIKSLIS} m)")

        info = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax_sl = plt.axes([0.2, 0.08, 0.6, 0.03])
        # Slaideri väärtus on nüüd samuti meetrites
        slaider = Slider(ax_sl, 'Mõõdupuu (m)', 1.0, 5000.0, valinit=METREID_PIKSLIS)

        def uuenda(val):
            # Teisendame slaideri meetrid tagasi pikslite tolerantsiks
            tol_px = slaider.val / METREID_PIKSLIS
            lihtne = algne_joon.simplify(tol_px, preserve_topology=True)
            
            if lihtne.geom_type.startswith('Multi'):
                x, y = [], []
                for g in lihtne.geoms:
                    lx, ly = g.xy
                    x.extend(lx); y.extend(ly)
            else:
                x, y = lihtne.xy
                
            joonis.set_data(x, y)
            
            pikkus_m = lihtne.length * METREID_PIKSLIS
            pikkus_km = pikkus_m / 1000
            
            info.set_text(f"Mõõdupuu pikkus: {int(slaider.val)} m\n"
                          f"Rannajoone pikkus: {int(pikkus_m)} m\n"
                          f"({pikkus_km:.2f} km)")
            fig.canvas.draw_idle()

        slaider.on_changed(uuenda)
        uuenda(METREID_PIKSLIS)
        plt.show()

    except Exception as e:
        print(f"Viga: {e}")

if __name__ == "__main__":
    kaima()