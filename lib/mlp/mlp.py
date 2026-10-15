import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as tfk
tfd = tfp.distributions


# ============================================================
# Multimodal MLP baseline - intermediate fusion
# m1 = EDA/PPG
# m2 = PPG/TEXT
# TEXT only in m2
#
# Same Modality objects as MVAE (encoder + decoder), but no VAE stage:
# each encoder gives a deterministic 50-d feature (posterior mean),
# features are concatenated here and classified, trained end-to-end with BCE.
# Decoders are not used by this model.
# ============================================================
class MLP(tfk.Model):
    def __init__(self,m1_modality,m2_modality,bert_lr=1e-6,name="MLP",**kwargs):
        super().__init__(name=name,**kwargs)

        self.m1      = m1_modality                                  # EDA/PPG
        self.m2      = m2_modality                                  # PPG/TEXT
        self.bert_lr = bert_lr                                      # BERT lr if text encoder trains BERT

        self.fused_dim  = self.m1.latent_dim + self.m2.latent_dim     # concatenated feature dimension
        self.classifier = tfk.layers.Dense(1)                        # binary classifier head

    def encode(self,modality,x,training=False):
        if not isinstance(x,dict):
            x = tf.cast(x,tf.float32)                                # physiological input

        out,_ = modality.qz_x(x,training=training)                  # Encoder / TextEncoder output

        if isinstance(out,tfd.Distribution):
            return out.mean()                                       # deterministic feature, same as MVAE classifier

        return out

    def fuse(self,h_m1,h_m2):
        return tf.concat([h_m1,h_m2],axis=-1)                       # batch, latent_m1 + latent_m2

    def call(self,inputs,y=None,training=False):
        h_m1 = self.encode(self.m1,inputs["m1"],training=training)
        h_m2 = self.encode(self.m2,inputs["m2"],training=training)

        h      = self.fuse(h_m1,h_m2)                                # intermediate fusion
        logits = self.classifier(h,training=training)

        if y is not None:
            y = tf.cast(y,tf.float32)
            y = tf.reshape(y,(-1,1))

            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y,
                logits=logits,
            )

            self.cls_loss = tf.reduce_mean(loss)
            self.loss     = self.cls_loss

        return logits

    @tf.function
    def train(self,inputs,y,optimizer):
        with tf.GradientTape() as tape:
            self.call(inputs,y=y,training=True)

        classifier_params = self.classifier.trainable_variables      # classifier head params
        encoder_params    = self.m1.encoder.params() + self.m2.encoder.params()  # encoder params only

        if hasattr(self.m2.encoder,"bert_params"):
            bert_params = self.m2.encoder.bert_params()              # only TextEncoder has this
        else:
            bert_params = []                                         # no BERT for PPG

        bert_ids         = set([id(v) for v in bert_params])
        encoder_non_bert = [v for v in encoder_params if id(v) not in bert_ids]  # remove BERT from encoder params

        params    = classifier_params + encoder_non_bert + bert_params
        gradients = tape.gradient(self.loss,params)

        n_cls = len(classifier_params)
        n_enc = len(encoder_non_bert)

        classifier_gradients = gradients[:n_cls]
        encoder_gradients    = gradients[n_cls:n_cls+n_enc]
        bert_gradients       = gradients[n_cls+n_enc:]

        if len(bert_params)>0:
            base_lr    = tf.cast(optimizer.learning_rate,tf.float32) # current optimizer lr
            bert_scale = tf.cast(self.bert_lr,tf.float32)/base_lr    # effective BERT lr = bert_lr

            bert_gradients = [
                None if g is None else g*bert_scale
                for g in bert_gradients
            ]                                                        # smaller BERT update

        final_gradients = classifier_gradients + encoder_gradients + bert_gradients

        optimizer.apply_gradients([
            (g,v) for g,v in zip(final_gradients,params) if g is not None
        ])

        return self.loss

    def predict_logits(self,inputs):
        logits = self.call(inputs,training=False)
        return tf.squeeze(logits,axis=-1)

    def predict_proba(self,inputs):
        logits = self.predict_logits(inputs)
        probs  = tf.sigmoid(logits)
        return probs

    def predict_class(self,inputs):
        probs = self.predict_proba(inputs)
        preds = tf.cast(probs >= 0.5,tf.int32)
        return preds
