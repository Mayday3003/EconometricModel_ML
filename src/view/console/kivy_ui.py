import pandas as pd
from sklearn.linear_model import LinearRegression
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import ObjectProperty

# Estructura del diseño con FileChooser
kv = '''
<RootWidget>:
    orientation: 'vertical'
    padding: 20
    spacing: 10

    Button:
        text: 'Cargar archivo train.csv'
        size_hint_y: None
        height: 50
        on_press: root.abrir_selector_archivos()

    Label:
        id: file_label
        text: 'Archivo no cargado.'
        size_hint_y: None
        height: 30

    BoxLayout:
        spacing: 10
        Label:
            text: 'Área habitable (pies cuadrados):'
        TextInput:
            id: area_input
            input_filter: 'float'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Año de construcción:'
        TextInput:
            id: year_input
            input_filter: 'int'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Habitaciones sobre el suelo:'
        TextInput:
            id: bedrooms_input
            input_filter: 'int'
            multiline: False

    Button:
        text: 'Predecir Precio'
        size_hint_y: None
        height: 50
        on_press: root.predecir_precio()

    Label:
        id: resultado_label
        text: ''
        font_size: 24
        halign: 'center'

<FileSelectorPopup>:
    title: 'Selecciona el archivo train.csv'
    size_hint: 0.9, 0.9
    auto_dismiss: False
    BoxLayout:
        orientation: 'vertical'
        FileChooserListView:
            id: filechooser
            path: '.'
        BoxLayout:
            size_hint_y: None
            height: 40
            spacing: 10
            padding: 5
            Button:
                text: 'Cancelar'
                on_press: root.dismiss()
            Button:
                text: 'Cargar'
                on_press: root.cargar_archivo(filechooser.path, filechooser.selection)
'''

class FileSelectorPopup(Popup):
    cargar_callback = ObjectProperty(None)

    def cargar_archivo(self, path, selection):
        if selection:
            archivo = selection[0]
            self.cargar_callback(archivo)
            self.dismiss()

class RootWidget(BoxLayout):
    modelo = None
    features = ["GrLivArea", "YearBuilt", "BedroomAbvGr"]

    def abrir_selector_archivos(self):
        popup = FileSelectorPopup(cargar_callback=self.cargar_csv)
        popup.open()

    def cargar_csv(self, ruta):
        try:
            df = pd.read_csv(ruta)
            df = df.dropna(subset=self.features + ["SalePrice"])
            X = df[self.features]
            y = df["SalePrice"]

            self.modelo = LinearRegression()
            self.modelo.fit(X, y)
            self.ids.file_label.text = f"Archivo cargado: {ruta.split('/')[-1]}"
            self.ids.resultado_label.text = "Modelo entrenado. Listo para predecir."
        except Exception as e:
            self.ids.resultado_label.text = f"Error cargando CSV: {e}"

    def predecir_precio(self):
        if self.modelo is None:
            self.ids.resultado_label.text = "Carga un archivo CSV primero."
            return

        try:
            area = float(self.ids.area_input.text)
            year = int(self.ids.year_input.text)
            bedrooms = int(self.ids.bedrooms_input.text)
            entrada = pd.DataFrame([[area, year, bedrooms]], columns=self.features)
            pred = self.modelo.predict(entrada)[0]
            self.ids.resultado_label.text = f"Precio estimado: ${pred:,.2f}"
        except Exception as e:
            self.ids.resultado_label.text = "Error en los datos ingresados."

class MainApp(App):
    def build(self):
        Builder.load_string(kv)
        return RootWidget()

if __name__ == '__main__':
    MainApp().run()
