from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout
from keras.models import Model
from keras import optimizers

def get_model():
    input_cat = Input((len(cat_features_hash), ))
    input_num = Input((len(num_features), ))
    
    input_tfidf = Input((size_tfidf, ), sparse=True)
    x_tfidf = Dense(100, activation="relu")(input_tfidf)
    x_tfidf = Dropout(0.5)(x_tfidf)
    
    x_cat = Embedding(max_size, 10)(input_cat)
    x_cat = Flatten()(x_cat)
    x_cat = Dropout(0.5)(x_cat)

    
    x_cat = Dense(100, activation="relu")(x_cat)

    x_num = Dense(100, activation="relu")(input_num)
    x_num = Dropout(0.5)(x_num)

    x = concatenate([x_cat, x_num, x_tfidf])

    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_tfidf], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.001, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = get_model()
model.fit([X_train_cat, X_train_num, X_train_tfidf], X_train_target, validation_split=0.1, epochs=5, batch_size=128)

pred_test = model.predict([X_test_cat, X_test_num, X_test_tfidf])
test["project_is_approved"] = pred_test
test[['id', 'project_is_approved']].to_csv("sub/lstm.csv", index=False)