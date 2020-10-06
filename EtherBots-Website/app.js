var express     =require("express");
var app         = express();
var bodyParser  = require("body-parser");
var methodOverride = require("method-override");


app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine","ejs");
app.use(express.static(__dirname + "/public"));
app.use(methodOverride("_method"));

app.get("/",function(req,res){
	res.render("index");
});

app.get("/home",function(req,res){
	res.render("home");
});

app.get("/stats",function(req,res){
	res.render("stats");
});


app.listen(process.env.PORT || 3000 ,process.env.IP,function(){
    console.log("Server has Started");
});