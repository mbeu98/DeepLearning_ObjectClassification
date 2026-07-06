import keras
from keras import ops

#Konstruktor setzt den Student und Teacher
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
    #Einstellen der Trainingsparameter
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha, temperature):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    #Berechnung des Distillation Loss durch die Logits vom Teacher und dem Loss des Students
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(ops.softmax(teacher_pred / self.temperature, axis=1),ops.softmax(y_pred / self.temperature, axis=1),) * (self.temperature**2)

        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        return total_loss

    def call(self, x):
        return self.student(x)