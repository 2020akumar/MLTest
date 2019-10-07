package com.example.mltest;

import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.Kwarg;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {
    private Python py;
    private Button mButton;
    private EditText mEdit;
    private EditText mEditConv;
    private EditText mEditDense;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Python.start(new AndroidPlatform(this));
        py=Python.getInstance();
//        py.getModule("main").callAttr("main");
        mButton=(Button)findViewById(R.id.button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                mButton.setText("Training Started");
//                PyObject y=py.getModule("main");
                String seconds= mEdit.getText().toString();
                String convs=mEditConv.getText().toString();
                String dense=mEditDense.getText().toString();
                int kw=Integer.parseInt(seconds);
                new trainTask().execute(kw,Integer.parseInt(convs),Integer.parseInt(dense));
//
//                PyObject x=y.callAttr("main",new Kwarg("second",kw));
//                mButton.setText(x.toString());
            }
        });
        mEdit=(EditText)findViewById(R.id.edit);
        mEditConv=(EditText)findViewById(R.id.editText);
        mEditDense=(EditText)findViewById(R.id.editText2);


    }
    private class trainTask extends AsyncTask<Integer,Void,Double>{
        @Override
        protected void onPreExecute(){
            super.onPreExecute();
            mButton.setText("Training Started");


        }
        @Override
        protected Double doInBackground(Integer... params){
                int time=params[0];
                int convols=params[1];
                int denses=params[2];
                PyObject y=py.getModule("main");



                PyObject x=y.callAttr("main",new Kwarg("second",time));
                Double acc=Double.parseDouble(x.toString());
                return acc;

        }
        @Override
        protected void onPostExecute(Double acc){
            super.onPostExecute(acc);
            mButton.setText(acc.toString());
        }
    }
}
