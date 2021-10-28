"""
Модуль для работы с ML-моделями.

API реализован при помощи библиотеки flask: декораторы route позволяют гибко
настроить поведение рассматриваемого модуля в соответствии с
HTTP-методом (POST, GET, PUT, DELETE) и указанным URL.

В основе модуля лежит класс MyModel, являющийся оберткой над ML-моделями
разных типов, образуя тем самым абстракцию «модель».
Обработка ошибок: методы класса проверяют корректность входных данных, не
корректируют ход выполнения программы, но выбрасывают сообщение об ошибке.

"""

# Импорт модулей flask и datetime (datetime исключительно для проверки работоспособности сервиса).
from flask import Flask, request, jsonify
from datetime import datetime

# Импорт используемых моделей.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

'''
Словарь, определяющий поддерживаемые модулем ML-модели.
Единообразие интерфейсов моделей sklearn (использование одинаковых методов)
и отсутствие необходимости квалифицировать класс экземпляра, для которого вызывается функция (fit, predict),
позволяют, в некотором смысле, реализовать концепцию наследования и описать
поведение Модели для всех ML-алгоритмов безотносительно их типа.
Таким образом, словарь содержит минимальную информацию: соответствие "текстовый код классификатора" - "имя класса".
'''
MODEL_TYPES = {
    'RandomForestClassifier': RandomForestClassifier,
    'RandomForestRegressor': RandomForestRegressor,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'ExtraTreesRegressor': ExtraTreesRegressor,
    'LogReg': LogisticRegression,
    'LinReg': LinearRegression
}

'''
Интерпретируемые описания ошибок выполнения (для единообразия собраны в один словарь).
'''
ERROR_MESSAGES = {
    'NEED_FIELD': '''Need {0} field''',
    'WRNG_UID': '''Wrong model`s UID ({0})''',
    'NOT_FIT': '''Model is not fitted'''
}


# Класс МОДЕЛЬ
class MyModel:
    def __init__(self):
        self.uid = None
        self.name = 'default_model_for_tests'
        self.tp = list(MODEL_TYPES.keys())[0]
        self.hyper_params = {'random_state': 0, }
        self.model_handle = None

    def update(self, params):
        """
        Метод позволяет обновить параметры (атрибуты класса (name, params)) модели.
        :param params: dict. Словарь, кодирующий новые значения параметров модели.
        """

        # Флаг "после обновления необходимо заново обучить модель".
        need_refit = False

        # Анализ переданных полей и предобработка (если нужно) значений.
        # Неизвестные поля игнорируются.
        for field in params:
            val = params[field]
            if field == 'name':
                self.name = val
                # Смена имени не требует обучения заново.
            elif (field == 'type') and (val in list(MODEL_TYPES.keys())):
                self.tp = val
                need_refit = True
            elif field == 'hyper_params':
                self.hyper_params = val
                need_refit = True

        # Удаление обученного классификатора, если необходимо.
        if need_refit:
            self.unfit()

    def get_uid(self):
        """
        Для разграничения атрибутов класса и их значения в рамках класса,
        а также контроля доступа к внутренней структуре экземпляра –
        рекомендуется использовать геттер, а не читать поле напрямую.
        :return: UID экземпляра
        """

        return self.uid

    def set_uid(self, uid):
        """
        Устанавливает UID экземпляра (сеттер).
        """

        self.uid = uid

    def have_clf(self):
        """
        :return: значение флага «у модели есть обученный классификатор»
        (другими словами, ранее к модели был корректно применён метод fit).
        """

        return self.model_handle is not None

    def get_info(self):
        """
        :return: dict. Cловарь, описывающий состояние модели; словарь содержит
        только интерпретируемые описания (то есть это необратимая сериализация экземпляра).
        """

        return {
            'uid': self.uid,
            'name': self.name,
            'type': self.tp,
            'hyper_params': self.hyper_params,
            'is_fitted': self.have_clf()
        }

    def fit(self, x, y):
        """
        Обучение модели.
        :param x: независимые переменные (аналогично sklearn).
        :param y: целевая переменная (аналогично sklearn).
        """

        '''
        Сохраняем дескриптор классификатора, вызывав (в качестве конструктора
        экземпляра) функцию, сопоставленную данному коду классификатора в
        словаре. При помощи механизма ** передаём все необходимые классификатору
        ГП из поля hyper_params.
        '''
        self.model_handle = MODEL_TYPES[self.tp](**self.hyper_params)
        self.model_handle.fit(x, y)

    def unfit(self):
        """
        Перезапись дескриптора модели состоянием None, т.е. сообщаем, что
        экземпляр без обученной модели (см. также метод have_clf).
        """

        self.model_handle = None

    def predict(self, x, proba):
        """
        Вернуть прогноз модели.
        :param x: входные независимые переменные.
        :param proba: выдать ли прогнозируемые вероятности (если True) или прогнозируемый класс (если False)
        (для классификаторов).
        :return: предсказание по модели.
        """

        if not self.have_clf():
            return None
        if proba:
            return self.model_handle.predict_proba(x)
        return self.model_handle.predict(x)


"""
Объект, соответствующий абстракции «совокупность моделей» обеспечивается
классом MyModelsList.
"""

# Класс СОВОКУПНОСТЬ МОДЕЛЕЙ
class MyModelsList:
    def __init__(self):
        # Положим в список одну модель сразу.
        self.items = [
            MyModel(),
        ]
        # Проставим UID-ы для предустановленных моделей.
        for ind, item in enumerate(self.items):
            item.set_uid(ind + 1)
        # Cчётчик моделей.
        self.counter = len(self.items)

    def get(self, uid):
        """
        :param uid: уникальный идентификатор модели.
        :return: экземпляр класса по его UID.
        """

        for item in self.items:
            if item.get_uid() == uid:
                return item
        return None

    def get_all(self):
        """
        :return: список описаний всех содержащихся в Совокупности моделей.
        """

        s = list()
        for item in self.items:
            s.append(item.get_info())
        return s

    def create(self, data):
        """
        Cоздаёт модель.
        :param data: dict. Cловарь, кодирующий новые значения атрибутов класса модели (name, type, hyper_params).
        :return: состояние модели (см. метод get_info).
        """

        self.counter += 1
        model = MyModel()
        model.set_uid(self.counter)
        model.update(data)
        self.items.append(model)
        return model.get_info()

    def update(self, uid, data):
        """
        Обновляет параметры (в широком смысле) модели.
        :param uid:
        :param data: dict. Cловарь, кодирующий новые значения атрибутов класса модели (name, type, hyper_params).
        :return: состояние модели (см. метод get_info).
        """

        item = self.get(uid)
        if item is None:
            return item
        item.update(data)
        return item.get_info()

    def delete(self, uid):
        """
        Удаляет модель.
        :param uid: уникальный идентификатор модели.
        :return: удалось или нет удалить модель (если попытка удалить несуществующую модель - False, иначе - True)
        """

        item = self.get(uid)
        if item is None:
            return False
        self.items.remove(item)
        return True


app = Flask(__name__)

models_base = MyModelsList()

'''
Helper-function для проверки наличия ключа в словаре и генерации сообщения об ошибке, если ключа не окажется.
'''


def check_field_in_dict_errmsg(d, name):
    if name not in list(d.keys()):
        return ERROR_MESSAGES['NEED_FIELD'].format(name)
    return ''  # пустая строка - ключ есть


# ------------------ API ------------------
'''
/test - проверка работоспособности сервиса
/classes - доступные типы алгоритмов
/models
    POST - создать модель
    GET - получить список моделей
/models/z
    PUT - обновить параметры модели z
    DELETE - удалить модель z
/models/z/fit (PUT) - обучить модель z
/models/z/predict (POST) - прогноз моделью z

В связи со спецификой json, все передаваемые данные приводятся к int, float, str
в соответствии с комментариями к запросам.
Во входных данных матрицы представляют собой список строк, в свою очередь каждая
строка – список значений элементов.
Прогноз модели (выходные данные) для универсальности вытягивается в одномерный
массив, который может быть восстановлен на принимающей стороне при помощи вызова
np.reshape (для чего также возвращаются исходные размеры массива).
'''


@app.route('/test')
def do_some() -> str:
    """
    # Запрос /test позволяет проверить работоспособность сервиса.
    """

    return 'app is run! Server\'s time is [' + str(datetime.now()) + ']'


@app.route('/classes', methods=['GET', ])
def show_models_type():
    """
    Запрос /classes вернёт список доступных типов классификаторов.
    """

    return jsonify(list(MODEL_TYPES.keys()))


@app.route('/models', methods=['POST', 'GET', ])
def model_read_or_add():
    """
    Запрос /models для метода POST создаст новую модель
    (с параметрами из json запроса) и вернёт её описание;

    для метода GET вернёт список моделей в Совокупности.
    """

    if request.method == 'GET':
        return jsonify(models_base.get_all())
    elif request.method == 'POST':
        return jsonify(models_base.create(request.get_json()))


@app.route('/models/<int:uid>', methods=['PUT', 'DELETE'])
def model_upd_or_del(uid):
    """
    Запрос /models/z
    для метода PUT обновит параметры модели с UID=z (параметры - из json
    запроса) и вернёт её новое описание или пустоту, если модели с таким UID не
    существует.

    для метода DELETE удалит модель с UID=z и вернёт "True" (или "False" если
    модели с таким UID не существует).
    """

    if request.method == 'PUT':
        return jsonify(models_base.update(uid, request.get_json()))
    elif request.method == 'DELETE':
        return jsonify(str(models_base.delete(uid)))


@app.route('/models/<int:uid>/fit', methods=['PUT', ])
def model_fit(uid):
    """
    Запрос /models/z/fit (только метод PUT)
    Обучает модель с UID=z и возвращает описание ошибки или сообщение об успехе.
    Структура json запроса:
    {
    'x':float список строк (списков значений) - факторы
    'y':int список меток класса
    }
    """

    model = models_base.get(uid)

    if model is None:
        return ERROR_MESSAGES['WRNG_UID'].format(uid)

    model.unfit()
    data = request.get_json()

    s = check_field_in_dict_errmsg(data, 'x')
    if s != '':
        return s

    s = check_field_in_dict_errmsg(data, 'y')
    if s != '':
        return s

    model.fit(data['x'], data['y'])
    return 'fitted model with uid ' + str(uid)


@app.route('/models/<int:uid>/predict', methods=['POST', ])
def model_predict(uid):
    """
    Запрос /models/z/predict (только метод POST)
    Возвращает прогноз моделью с UID=z или описание ошибки.
    Структура json запроса:
    {
    'x':float факторы (аналогично запросу /models/z/fit)
    'proba':int флаг "использовать predict_proba" (иначе - predict)
    }
    Структура возвращаемого json:
    {
    'x':float прогноз, развёрнутый в 1D список
    'shape':int исходные размеры x
    }
    """

    model = models_base.get(uid)

    if model is None:
        return ERROR_MESSAGES['WRNG_UID'].format(uid)

    if not model.have_clf():
        return ERROR_MESSAGES['NOT_FIT']

    data = request.get_json()

    s = check_field_in_dict_errmsg(data, 'x')
    if s != '':
        return s

    s = check_field_in_dict_errmsg(data, 'proba')
    if s != '':
        return s

    preds = model.predict(data['x'], data['proba'] == 1)
    d = {
        'x': [float(v) for v in preds.reshape(-1)],
        'shape': preds.shape
    }

    return jsonify(d)


# Для запуска при локальном тестировании (при выполнении тела файла).
if __name__ == '__main__':
    app.run(debug=True)
