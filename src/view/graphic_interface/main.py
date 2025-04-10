import sys
import os
import pandas as pd
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Imports del pipeline existente
from controller.data_controller import engineer_features, divide_dataframes
from model.model_training import train_model

# Kivy interface layout
kv = '''
<MainLayout>:
    orientation: 'vertical'
    padding: 20
    spacing: 10

    Button:
        text: 'Load train.csv file'
        size_hint_y: None
        height: 50
        on_press: root.open_file_chooser()

    Label:
        id: file_status
        text: 'No file loaded.'
        size_hint_y: None
        height: 30

    BoxLayout:
        spacing: 10
        Label:
            text: 'Living Area (GrLivArea):'
        TextInput:
            id: input_area
            input_filter: 'float'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Year Built (YearBuilt):'
        TextInput:
            id: input_year
            input_filter: 'int'
            multiline: False

    BoxLayout:
        spacing: 10
        Label:
            text: 'Bedrooms Above Ground (BedroomAbvGr):'
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
    selected_features = ["grlivarea", "yearbuilt", "bedroomabvgr"]
    trained_columns = None

    def open_file_chooser(self):
        FileChooserPopup(load_callback=self.process_training_file).open()

    def process_training_file(self, file_path):
        try:
            # Prepare dataset using project's pipeline
            target_column = "saleprice"
            bins_labels = ([0, 100000, 200000, 300000, 450000, 760000], [0, 1, 2, 3, 4])

            df_train, df_val, df_test, y_train, _, _, full_data = divide_dataframes(
                file_path, target_column, bins_labels
            )

            df_train, df_val, df_test = engineer_features(
                (df_train, df_val, df_test), full_data,
                ['neighborhood', 'exterior2nd', 'housestyle']
            )

            self.trained_columns = df_train.columns  # Save structure for prediction input
            self.model = train_model(df_train, y_train)

            filename = os.path.basename(file_path)
            self.ids.file_status.text = f"File loaded: {filename}"
            self.ids.result_label.text = "Model trained. Ready to predict."
        except Exception as error:
            self.ids.result_label.text = f"Error loading file: {str(error)}"

    def predict_price(self):
        if self.model is None or self.trained_columns is None:
            self.ids.result_label.text = "Please load and train the model first."
            return

        try:
            grlivarea = float(self.ids.input_area.text)
            yearbuilt = int(self.ids.input_year.text)
            bedroomabvgr = int(self.ids.input_bedrooms.text)

            prediction = self.make_prediction(grlivarea, yearbuilt, bedroomabvgr)
            self.ids.result_label.text = f"Estimated Price: ${prediction:,.2f}"
        except Exception:
            self.ids.result_label.text = "Invalid input. Please check values."

    def make_prediction(self, area, year, bedrooms):
        # Construct full input with default values and user input
        input_data = pd.DataFrame([0], columns=["dummy"])
        input_data = pd.DataFrame(columns=self.trained_columns)
        input_data.loc[0] = 0  # Fill with zeroes
        input_data.loc[0, "grlivarea"] = area
        input_data.loc[0, "yearbuilt"] = year
        input_data.loc[0, "bedroomabvgr"] = bedrooms

        return self.model.predict(input_data)[0]

class HousePriceApp(App):
    def build(self):
        Builder.load_string(kv)
        return MainLayout()

if __name__ == '__main__':
    HousePriceApp().run()
