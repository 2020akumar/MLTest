package com.example.mltest;

import androidx.appcompat.app.AppCompatActivity;
import com.chaquo.python.Kwarg;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import android.annotation.TargetApi;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    private Python py;
    private Button mButton;
    private EditText mEdit;
    private EditText mEditConv;
    private EditText mEditDense;
    private TextView mRes;
    private ProgressBar mTrainPercent;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Python.start(new AndroidPlatform(this));
        py=Python.getInstance();
        mRes=(TextView)findViewById(R.id.res);
        mTrainPercent=(ProgressBar)findViewById(R.id.trainPercent);
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
        File file =new File("model");
        String x=file.getAbsolutePath();
        String y=file.getPath();


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
//                PyObject y=py.getModule("main");



                PyObject x=py.getModule("main");
                PyObject   y=x.callAttr("main",new Kwarg("second",time), new Kwarg("conv",convols), new Kwarg("dens",denses));

                Double acc=Double.parseDouble(y.toString())*100.0;
                //x.callAttr("run","");
                x.close();
                y.close();
                return acc;

        }
        @Override
        @TargetApi(24)
        protected void onPostExecute(Double acc){
            super.onPostExecute(acc);
            String fin=acc.toString();
            if (fin.length()>5){
                fin=fin.substring(0,6);
            }
            mRes.setText(fin+"%");

            mTrainPercent.setProgress((int)(Double.parseDouble(fin)));
            mButton.setText("Train Again");
        }
    }
}
