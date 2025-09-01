def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict(client):
    response = client.post(
        '/predict', json={'file_path': 'tests/fake_audio.wav'}
    )
    assert response.status_code == 200
    assert response.json() == {
        'digit': 0,
        'logits': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_predict_invalid_file(client):
    response = client.post(
        '/predict', json={'file_path': 'tests/fake_audio.txt'}
    )
    assert response.status_code == 422
    assert response.json() == {
        'detail': [
            {
                'ctx': {
                    'error': {},
                },
                'input': 'tests/fake_audio.txt',
                'loc': [
                    'body',
                    'file_path',
                ],
                'msg': 'Value error, file_path must point to a .wav file',
                'type': 'value_error',
            },
        ],
    }
