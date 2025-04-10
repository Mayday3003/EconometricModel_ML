import pandas as pd
from sklearn.linear_model import LinearRegression

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty

kv = '''
<MainLayout>:
    orientation: 'vertical'
    padding: 20
    spacing: 10

    Button:
        text: 'Load train.csv file'
        size_hint_y: None
        height: 50
        on_press: root.show_file_dialog()

    Label:
        id: file_status
        text: 'No file loaded.'
        size_hint_y: None
        height: 30

    BoxLayout:
        spacing: 10
        Label:
            text: 'Living Area (sq ft):'
        TextInput:
            id: input_area
            input_filter: 'float'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Year Built:'
        TextInput:
            id: input_year
            input_filter: 'int'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Bedrooms Above Ground:'
        TextInput:
            id: input_bedrooms
            input_filter: 'int'
            multiline: False

    Button:
        text: 'Predict Price'
        size_hint_y: None
        height: 50
        on_press: root.predict_price()

    Label:
        id: result_label
        text: ''
        font_size: 24
        halign: 'center'

<FileChooserPopup>:
    title: 'Select train.csv file'
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
                text: 'Cancel'
                on_press: root.dismiss()

            Button:
                text: 'Load'
                on_press: root.load_file(filechooser.selection)
'''

class FileChooserPopup(Popup):
    load_callback = ObjectProperty(None)

    def load_file(self, selection):
        if selection:
            self.load_callback(selection[0])
            self.dismiss()

class MainLayout(BoxLayout):
    model = None
    feature_columns = ["GrLivArea", "YearBuilt", "BedroomAbvGr"]

    def show_file_dialog(self):
        FileChooserPopup(load_callback=self.load_csv).open()

    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            required_columns = self.feature_columns + ["SalePrice"]

            if not all(col in df.columns for col in required_columns):
                self.ids.result_label.text = "Missing required columns."
                return

            df = df.dropna(subset=required_columns)
            X = df[self.feature_columns]
            y = df["SalePrice"]

            self.model = LinearRegression()
            self.model.fit(X, y)

            file_name = file_path.split("/")[-1]
            self.ids.file_status.text = f"File loaded: {file_name}"
            self.ids.result_label.text = "Model trained. Ready to predict."
        except Exception as e:
            self.ids.result_label.text = f"Error: {str(e)}"

    def predict_price(self):
        if self.model is None:
            self.ids.result_label.text = "Please load a CSV file first."
            return

        try:
            area = float(self.ids.input_area.text)
            year = int(self.ids.input_year.text)
            bedrooms = int(self.ids.input_bedrooms.text)
            price = self.make_prediction(area, year, bedrooms)
            self.ids.result_label.text = f"Estimated Price: ${price:,.2f}"
        except Exception:
            self.ids.result_label.text = "Invalid input. Check the values."

    def make_prediction(self, area, year, bedrooms):
        input_df = pd.DataFrame([[area, year, bedrooms]], columns=self.feature_columns)
        return self.model.predict(input_df)[0]

class HousePricePredictorApp(App):
    def build(self):
        Builder.load_string(kv)
        return MainLayout()

if __name__ == '__main__':
    HousePricePredictorApp().run()
